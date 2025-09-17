import jax
import jax.numpy as jnp
import inspeqtor.experimental as sq
from functools import partial
import typing
import pathlib
from rich.progress import track
import pandas as pd
import typer
from rich.console import Console
from rich.prompt import Prompt, IntPrompt, FloatPrompt
import numpy as np

sq.utils.enable_jax_x64()

app = typer.Typer()


def get_PSD_v1() -> typing.Callable[[jnp.ndarray], jnp.ndarray]:
    alpha = 1
    beta = 0.8
    peak = 15

    def spectrum(freqency):
        return (1 / (freqency + 1) ** (alpha)) + (
            beta * jnp.exp(-((freqency - peak) ** 2) / 10)
        )

    return spectrum


def make_noise_fn(
    power_spectrum: jnp.ndarray,
    frequency_space: jnp.ndarray,
    time_step: int,
    dw: float,
):
    n_frequency = frequency_space.shape[0]
    n_dimensions = 1

    def noise_fn(key: jnp.ndarray):
        phi = 2 * jnp.pi * jax.random.uniform(key, shape=(n_frequency,))
        fourier_coefficient = jnp.exp(phi * 1.0j) * jnp.sqrt(
            2 ** (n_dimensions + 1) * power_spectrum * dw
        )
        samples = jnp.fft.fftn(fourier_coefficient, s=(time_step,))
        return jnp.real(samples)

    return noise_fn


def make_noisy_signal_fn(
    signal_fn,
    t: jnp.ndarray,
    spectrum: jnp.ndarray,
    frequency_space: jnp.ndarray,
    noise_str: float,
    dw: float,
):
    noise_fn = make_noise_fn(spectrum, frequency_space, t.shape[0], dw)

    def param_to_noisy_signal(key, params):
        return signal_fn(params, t) + noise_str * noise_fn(key)

    return param_to_noisy_signal


def make_final_returned_whitebox_fn(
    stochastic_whitebox: typing.Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
):
    def final_returned_whitebox(key: jnp.ndarray, params: jnp.ndarray):
        return stochastic_whitebox(key, params)[-1]

    return final_returned_whitebox


def make_prepare_unitary_ensemble(
    final_returned_whitebox: typing.Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    N_UNITARY_SAMPLES=1000,
):
    def prepare_unitary_ensemble(key: jnp.ndarray, params: jnp.ndarray):
        # simulate it for N_UNITARY_SAMPLES times to get a distribution of final unitary
        unitaries = jax.vmap(final_returned_whitebox, in_axes=(0, None))(
            jax.random.split(key, N_UNITARY_SAMPLES), params
        )

        return unitaries

    return prepare_unitary_ensemble


def make_shot_sample_fn(unitaries: jnp.ndarray):
    def shot_sample(
        key: jnp.ndarray, initial_state: jnp.ndarray, observable: jnp.ndarray
    ):
        key, unitary_key, shot_key = jax.random.split(key, 3)
        unitary = jax.random.choice(unitary_key, unitaries)

        expval = sq.physics.calculate_exp(unitary, observable, initial_state)
        prob = sq.utils.expectation_value_to_prob_plus(expval)
        return jax.random.choice(
            shot_key, jnp.array([1, -1]), shape=(1,), p=jnp.array([prob, 1 - prob])
        )

    return shot_sample


def make_single_param_simulator(stochastic_whitebox):
    def single_param_simulator(
        key: jnp.ndarray, control_param: jnp.ndarray, shots: int
    ):
        stochastic_expvals = jnp.zeros((18,))

        for idx, exp in enumerate(sq.constant.default_expectation_values_order):
            key, whitebox_key, shots_key = jax.random.split(key, 3)

            # This is slow
            stochastic_unitaries = jax.vmap(stochastic_whitebox, in_axes=(0, None))(
                jax.random.split(whitebox_key, shots), control_param
            )

            expval = jax.vmap(
                sq.utils.calculate_shots_expectation_value,
                in_axes=(0, None, 0, None, None),
            )(
                jax.random.split(shots_key, shots),
                exp.initial_density_matrix,
                stochastic_unitaries[:, -1, :, :],
                sq.constant.plus_projectors[exp.observable],
                1,
            )

            stochastic_expvals = stochastic_expvals.at[idx].set(jnp.mean(expval))

        return stochastic_expvals

    return single_param_simulator


def make_stochastic_trotterization_solver(
    hamiltonian: typing.Callable[..., jnp.ndarray],
    time_step: jnp.ndarray,
    y0: jnp.ndarray,
):
    """Retutn whitebox function compute using Trotterization strategy.

    Args:
        hamiltonian (typing.Callable[..., jnp.ndarray]): The Hamiltonian function of the system
        control_sequence (controlsequence): The pulse sequence instance
        dt (float, optional): The duration of time step in nanosecond. Defaults to 2/9.
        trotter_steps (int, optional): The number of trotterization step. Defaults to 1000.

    Returns:
        typing.Callable[..., jnp.ndarray]: Trotterization Whitebox function
    """
    hamiltonian = jax.jit(hamiltonian)

    def whitebox(key: jnp.ndarray, control_parameter: jnp.ndarray):
        hamiltonians = hamiltonian(key, control_parameter)

        unitaries = jax.scipy.linalg.expm(
            -1j * (time_step[1] - time_step[0]) * hamiltonians
        )
        # * Nice explanation of scan
        # * https://www.nelsontang.com/blog/a-friendly-introduction-to-scan-with-jax
        _, unitaries = jax.lax.scan(sq.physics.unitaries_prod, y0, unitaries)
        return unitaries

    return whitebox


def simulate_the_shot(key, probs):
    bin_key, prob_key = jax.random.split(key)
    prob = jax.random.choice(prob_key, probs)

    binary = jax.random.choice(
        bin_key, a=jnp.array([-1, 1]), shape=(), p=jnp.array([1 - prob, prob])
    )

    return binary


def make_noisy_hamiltonian(
    t: jnp.ndarray,
    qubit_info: sq.data.QubitInformation,
    noisy_signal_fn: typing.Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    detune: float,
):
    def rotating_transmon_hamiltonian(
        key: jnp.ndarray,
        params,
    ) -> jnp.ndarray:
        """Rotating frame hamiltonian of the transmon model

        Args:
            params (HamiltonianParameters): The parameter of the pulse for hamiltonian
            t (jnp.ndarray): The time to evaluate the Hamiltonian
            qubit_info (QubitInformation): The information of qubit
            signal (Callable[..., jnp.ndarray]): The pulse signal

        Returns:
            jnp.ndarray: The Hamiltonian
        """
        a0 = 2 * jnp.pi * qubit_info.frequency
        a1 = 2 * jnp.pi * qubit_info.drive_strength

        return (
            (
                a1
                * noisy_signal_fn(key, params).reshape(-1, 1, 1)
                * jnp.cos(a0 * t.reshape(-1, 1, 1))
                * sq.constant.X
            )
            - (
                a1
                * noisy_signal_fn(key, params).reshape(-1, 1, 1)
                * jnp.sin(a0 * t.reshape(-1, 1, 1))
                * sq.constant.Y
            )
            + detune * sq.constant.X
        )

    return rotating_transmon_hamiltonian


def make_sample_expectation_values(
    shot_sample: typing.Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
):
    def sample_expectation_values(key: jnp.ndarray, shots: int):
        stochastic_expvals = jnp.zeros((18,))
        for idx, exp in enumerate(sq.constant.default_expectation_values_order):
            key, subkey = jax.random.split(key)
            expval = jax.vmap(shot_sample, (0, None, None))(
                jax.random.split(subkey, shots),
                exp.initial_density_matrix,
                exp.observable_matrix,
            ).mean()
            stochastic_expvals = stochastic_expvals.at[idx].set(expval)
        return stochastic_expvals

    return sample_expectation_values


def quantum_device(
    key: jnp.ndarray, params: jnp.ndarray, shots: int, stochastic_solver
):
    unitaries_key, sampler_key = jax.random.split(key)
    # This is an expensive computation.
    unitaries = make_prepare_unitary_ensemble(stochastic_solver)(unitaries_key, params)
    # So, we use bootstrap method to save recomputation of unitary
    shot_sample_fn = make_shot_sample_fn(unitaries)
    sampler = make_sample_expectation_values(shot_sample_fn)
    # Sample the expectation value from the ensemble
    return sampler(sampler_key, shots)


def get_data_model(noise_str: float, detune: float, trotter_steps: int):
    qubit_info = sq.predefined.get_mock_qubit_information()
    control_sequence = sq.predefined.get_gaussian_control_sequence(
        qubit_info=qubit_info
    )
    device_cycle = 2 / 9

    signal_fn = sq.physics.signal_func_v5(
        get_envelope=sq.control.get_envelope_transformer(
            control_sequence=control_sequence
        ),
        drive_frequency=qubit_info.frequency,
        dt=device_cycle,
    )

    T = control_sequence.total_dt * device_cycle  # Time(1 / T = dw)
    nt = trotter_steps  # Num.of Discretized Time
    F = 1 / T * nt / 2  # Frequency.(Hz)
    nw = int(nt / 2)  # Num of Discretized Freq.

    # Generation of Input Data(Stationary)
    # dt = T / nt
    # time in nanosecond
    t = jnp.linspace(0, T, nt)
    dw = F / nw
    w = jnp.linspace(0, F, nw)

    S = get_PSD_v1()(w)

    y0 = jnp.eye(2, dtype=jnp.complex128)

    noisy_signal_fn = make_noisy_signal_fn(
        signal_fn, t, S, w, noise_str=noise_str, dw=dw
    )
    hamiltonian = make_noisy_hamiltonian(t, qubit_info, noisy_signal_fn, detune=detune)
    stochastic_solver = make_stochastic_trotterization_solver(hamiltonian, t, y0)

    ideal_hamiltonian = partial(
        sq.predefined.rotating_transmon_hamiltonian,
        qubit_info=qubit_info,
        signal=sq.physics.signal_func_v5(
            get_envelope=sq.control.get_envelope_transformer(
                control_sequence=control_sequence
            ),
            drive_frequency=qubit_info.frequency,
            dt=device_cycle,
        ),
    )

    whitebox = sq.physics.make_trotterization_solver(
        ideal_hamiltonian,
        control_sequence=control_sequence,
        dt=device_cycle,
        trotter_steps=trotter_steps,
        y0=y0,
    )

    return sq.utils.SyntheticDataModel(
        control_sequence=control_sequence,
        qubit_information=qubit_info,
        dt=device_cycle,
        solver=stochastic_solver,
        whitebox=whitebox,
        ideal_hamiltonian=ideal_hamiltonian,
        total_hamiltonian=hamiltonian,
        quantum_device=partial(
            quantum_device,
            stochastic_solver=make_final_returned_whitebox_fn(stochastic_solver),
        ),
    )


@app.command("dataset")
def generate_dataset():
    console = Console()

    identifier = Prompt.ask("🧐 Please give an identifier to the experiment")
    noise_str = FloatPrompt.ask("🔊 Noise strength", default=0.01)
    DETUNE = FloatPrompt.ask("🌊 Detune in X-axis", default=0.001)
    N_UNITARY_SAMPLES = IntPrompt.ask("🎲 Number of trajectory", default=1000)
    sample_size = IntPrompt.ask("💡 Sample size", default=1000)

    # Create path.
    path = pathlib.Path(f"./data/PSD/{identifier}")
    path.mkdir(parents=True, exist_ok=True)

    interm_expvals_path = path / "interm_expvals"
    interm_expvals_path.mkdir(parents=True, exist_ok=True)

    TROTTER_STEPS = 10_000

    data_model = get_data_model(
        noise_str=noise_str, trotter_steps=TROTTER_STEPS, detune=DETUNE
    )

    control_sequence = data_model.control_sequence

    parameter_structure = control_sequence.get_parameter_names()
    config = sq.data.ExperimentConfiguration(
        qubits=[data_model.qubit_information],
        expectation_values_order=sq.constant.default_expectation_values_order,
        parameter_names=parameter_structure,
        backend_name="trotterization_simulator",
        shots=1000,
        EXPERIMENT_IDENTIFIER=identifier,
        EXPERIMENT_TAGS=["paper_2", "PSD", "simulation"],
        device_cycle_time_ns=data_model.dt,
        description="",
        sequence_duration_dt=control_sequence.total_dt,
        instance="inspeqtor",
        sample_size=sample_size,
        additional_info={
            "NOISE_STR": noise_str,
            "TROTTERIZATION": True,
            "TROTTER_STEPS": TROTTER_STEPS,
            "DETUNE": DETUNE,
        },
    )

    console.log(config, markup=True, highlight=True)

    key = jax.random.key(0)

    pulse_params_list = []
    control_parameters = jnp.linspace(0.0, 2 * jnp.pi, config.sample_size).reshape(
        -1, 1
    )
    a2l_fn, _ = sq.control.get_param_array_converter(control_sequence=control_sequence)
    for sample_idx in range(config.sample_size):
        pulse_params_list.append(a2l_fn(control_parameters[sample_idx]))

    get_unitary_ensemble = make_prepare_unitary_ensemble(
        final_returned_whitebox=make_final_returned_whitebox_fn(data_model.solver),
        N_UNITARY_SAMPLES=N_UNITARY_SAMPLES,
    )

    console.log("Defined the control parameters")

    rows = []
    for sample_idx in track(range(config.sample_size), description="Processing..."):
        key, unitary_key, sample_key = jax.random.split(key, 3)

        # Generate the ensemble of unitary operators.
        unitary_ensemble = get_unitary_ensemble(
            unitary_key, control_parameters[sample_idx]
        )

        assert unitary_ensemble.shape == (N_UNITARY_SAMPLES, 2, 2)

        # Sample for the expectation values.
        expectation_values = make_sample_expectation_values(
            make_shot_sample_fn(unitary_ensemble)
        )(sample_key, config.shots)

        assert expectation_values.shape == (18,)

        interm_expvals = {}
        for exp_idx, exp in enumerate(sq.constant.default_expectation_values_order):
            # Calculate the statistic of the ensemble
            intermediate_expval = sq.physics.calculate_exp(
                unitary_ensemble,
                operator=exp.observable_matrix,
                density_matrix=exp.initial_density_matrix,
            )

            interm_expvals[f"{exp.initial_state}/{exp.observable}"] = np.array(
                intermediate_expval
            )

            # Save the experimental data
            row = sq.data.make_row(
                expectation_value=float(expectation_values[exp_idx]),
                initial_state=exp.initial_state,
                observable=exp.observable,
                parameters_list=pulse_params_list[sample_idx],
                parameters_id=sample_idx,
            )

            rows.append(row)

        interm_df = pd.DataFrame(interm_expvals)
        interm_df.to_csv(
            interm_expvals_path / f"parameters_{str(sample_idx).zfill(4)}.csv",
            index=False,
        )

    df = pd.DataFrame(rows)

    # ensemble_statistic_df = pd.DataFrame(ensemble_statistic_rows)
    exp_data = sq.data.ExperimentData(experiment_config=config, preprocess_data=df)

    sq.predefined.save_data_to_path(path, exp_data, control_sequence)
    # ensemble_statistic_df.to_csv(path / "ensemble_statistic.csv", index=False)

    console.log("Saved the dataset")
    console.log("Saved the ensemble of the unitary operators")


def feature_map(x: jnp.ndarray) -> jnp.ndarray:
    return sq.predefined.polynomial_feature_map(x / (2 * jnp.pi), degree=4)


if __name__ == "__main__":
    app()
