from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import networkx as nx
from enum import Enum
import numpy as np
import random

# Define support strengths and states of depression
SUPPORT_STRENGTHS = {
    'therapist': 5,
    'family': 2,
    'friend': 1
}


class State(Enum):
    MILD = 0
    MODERATE = 1
    SEVERE = 2


# Agent representing an individual
class Individual(Agent):
    def __init__(self, unique_id, model, depression_level, resilience_factor):
        super().__init__(unique_id, model)
        self.depression_level = depression_level
        self.resilience_factor = resilience_factor
        self.support_network = {'therapist': 0, 'family': 0, 'friend': 0}

    def step(self):
        net_support = 0
        print(f"Agent {self.unique_id} stepping. Current depression: {self.depression_level}")

        # Calculate support from neighbors
        for neighbor in self.model.G.neighbors(self.unique_id):
            neighbor_agent = self.model.G.nodes[neighbor]['agent']
            if isinstance(neighbor_agent, SupportiveMember) and random.random() < 0.5:
                support_type = neighbor_agent.member_type
                support_value = SUPPORT_STRENGTHS[support_type] * random.uniform(0.5, 1.5)

                # Determine if the individual accepts the help
                if random.random() < self.model.acceptance_probability:
                    net_support += support_value
                    print(f"Agent {self.unique_id} accepted {support_value:.2f} support from {support_type}")
                else:
                    # Determine the outcome if the individual refuses help
                    if random.random() < self.model.refusal_gets_worse_probability:
                        net_support -= support_value 
                        print(f"Agent {self.unique_id} refused help and got worse by {support_value:.2f}")
                    else:
                        net_support += support_value
                        print(f"Agent {self.unique_id} refused help but still got better by {support_value:.2f}")

        new_depression_index = self.depression_level.value - (net_support * self.model.support_effectiveness)
        print(f"Agent {self.unique_id} - Old Depression: {self.depression_level}, Net Support: {net_support}, New Depression Index: {new_depression_index}")

        if new_depression_index < 0.5:
            self.depression_level = State.MILD
        elif new_depression_index < 1.5:
            self.depression_level = State.MODERATE
        else:
            self.depression_level = State.SEVERE

        print(f"Agent {self.unique_id} - New Depression: {self.depression_level}")


# Agent representing a supportive member
class SupportiveMember(Agent):
    def __init__(self, unique_id, model, member_type):
        super().__init__(unique_id, model)
        self.member_type = member_type

    def step(self):
        pass


def compute_average_depression(model):
    # Gather all individuals' depression levels
    depression_levels = [agent.depression_level for agent in model.schedule.agents if isinstance(agent, Individual)]
    
    
    numeric_levels = [state.value for state in depression_levels]

    
    if len(numeric_levels) > 0:
        average_depression = sum(numeric_levels) / len(numeric_levels)
    else:
        average_depression = 0

    return average_depression

def count_state(model, state):
    return sum(1 for agent in model.schedule.agents if isinstance(agent, Individual) and agent.depression_level == state)


class DepressionSupportModel(Model):
    def __init__(self, num_agents, num_supportive, stigma_level, support_effectiveness, acceptance_probability, refusal_gets_worse_probability):
        super().__init__()
        self.num_agents = num_agents
        self.num_supportive = num_supportive
        self.stigma_level = stigma_level
        self.support_effectiveness = support_effectiveness
        self.acceptance_probability = acceptance_probability
        self.refusal_gets_worse_probability = refusal_gets_worse_probability
        self.G = nx.erdos_renyi_graph(n=num_agents + num_supportive, p=0.1)
        self.schedule = RandomActivation(self)

        print(f"Initializing model with {num_agents} agents and {num_supportive} supportive members.")

        for i in range(num_agents):
            depression_level = State(random.randint(0, 2))
            resilience_factor = random.uniform(0.1, 1.0)
            agent = Individual(i, self, depression_level, resilience_factor)
            self.G.nodes[i]['agent'] = agent
            self.schedule.add(agent)
            print(f"Added Individual agent {i} with initial depression level {depression_level}")

        for i in range(num_agents, num_agents + num_supportive):
            member_type = random.choice(list(SUPPORT_STRENGTHS.keys()))
            agent = SupportiveMember(i, self, member_type)
            self.G.nodes[i]['agent'] = agent
            self.schedule.add(agent)
            print(f"Added SupportiveMember agent {i} with type {member_type}")

        self.datacollector = DataCollector(
            model_reporters={
                "Average Depression": compute_average_depression,
                "Mild Count": lambda m: count_state(m, State.MILD),
                "Moderate Count": lambda m: count_state(m, State.MODERATE),
                "Severe Count": lambda m: count_state(m, State.SEVERE),
            }
        )

    def step(self):
        print("Model stepping.")
        self.schedule.step()
        self.datacollector.collect(self)
