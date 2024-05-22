from model import DepressionSupportModel, State, Individual

def test_model():
    num_agents = 50
    num_supportive = 10
    stigma_level = 0.3
    support_effectiveness = 0.5

    model = DepressionSupportModel(num_agents, num_supportive, stigma_level, support_effectiveness)

    for i in range(10):
        print(f"Step {i+1}")
        model.step()
        
        for agent in model.schedule.agents:
            if isinstance(agent, Individual):
                print(f"Agent {agent.unique_id} - Depression Level: {agent.depression_level}")

if __name__ == "__main__":
    test_model()
