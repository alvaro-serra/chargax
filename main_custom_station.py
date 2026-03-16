import jax
import jax.numpy as jnp
import jaxnasium as jym
import numpy as np
from jaxnasium.algorithms import DQN, PPO, SAC
from jaxtyping import Array, PRNGKeyArray

from chargax import EVSE, Chargax, ChargingStation, StationSplitter, StationBattery
from chargax.baselines import MaxCharge, Random

if __name__ == "__main__":
    rng = jax.random.PRNGKey(42)
    n_stations = 2 # 1/2/4
    n_batteries = 10 # 0/1/2/3
    # Initialize a default charging station environment from template:
    # charging_station = ChargingStation.init_default_station()
    charging_station = ChargingStation(
        max_kw_throughput=300.0,  # Grid connection limit
        efficiency=1.0,
        connections=[
            StationSplitter(
                max_kw_throughput=150.0*n_stations*2,
                efficiency=0.995,
                # 2 Fast chargers:
                connections=[
                    StationSplitter(
                        max_kw_throughput=150.0*n_stations*2,
                        efficiency=0.995,
                        connections=[EVSE(num_chargers=2, voltage=600, max_current=250, efficiency=0.995)
                                    for _ in range(n_stations)]
                    ),
                    # on-site batteries
                    StationBattery(
                        capacity_kw=500.0*n_batteries,
                        max_kw_throughput=150.0*n_batteries,
                        efficiency=0.995,
                    )
                ]
            )]

    )

    # Create the environment
    env = Chargax(station=charging_station,
                  allow_discharging=False
                  )
    env = jym.LogWrapper(env)

    # # RL Training with PPO
    # total_timesteps = 500_000
    # agent = PPO(  # Not optimized, just a simple example
    #     num_steps=300,
    #     num_envs=8,
    #     total_timesteps=total_timesteps,
    #     learning_rate=2.5e-4,
    #     anneal_learning_rate=True,
    #     normalize_rewards=False,
    #     normalize_observations=True,  # Important
    # )
    #
    # agent = agent.train(rng, env)
    #
    # results = agent.evaluate(rng, env, num_eval_episodes=25)
    # print(f"PPO - Average reward over 25 evaluation episodes: {np.mean(results)}")

    # Compare against baselines:
    print("Evaluating baselines...")
    rewards, profits = MaxCharge(env).evaluate(rng, num_eval_episodes=25)
    print(
        f"MaxCharge - Average cumulative reward: {np.sum(rewards, axis=1).mean():.2f}"
    )
    rewards, profits = Random(env).evaluate(rng, num_eval_episodes=25)
    print(f"Random - Average cumulative reward: {np.sum(rewards, axis=1).mean():.2f}")
