import numpy as np
from CG_source import Agent  # Assuming your Agent class is in 'CG_source.py'

def analyze_p_send_for_beta(state_vector, beta_values):
    """
    Calculates and prints the p_send vector (to 3 s.f.) for a given state vector
    across a range of beta (rationality) values.

    Args:
      state_vector: A list or numpy array representing the agent's beliefs.
      beta_values: A list of beta values to analyze.
    """

    # Create a dummy agent (we only need its p_send calculation logic)
    agent = Agent(
        name="DummyAgent",
        a=0.5,  # These values don't matter for p_send
        alpha=0.2,
        bi=1.0,  # Initial bi, we'll modify it later
        bj=1.0,
        eps=0.1,
        state_vector=np.array(state_vector),
        local_similarity=0.0
    )

    for beta in beta_values:
        agent.bi = beta  # Update the agent's rationality
        agent.update_probabilities()  # Recalculate p_send
        p_send_rounded = np.round(agent.p_send, 3)  # Round to 3 s.f.
        print(f"Beta = {beta:.2f}, p_send = {p_send_rounded}")

if __name__ == "__main__":
    # Example usage:
    state_vector = [0.8,-0.8]  # Define your state vector here
    beta_values =  [1,2,3,4,5,6,7]

    analyze_p_send_for_beta(state_vector, beta_values)