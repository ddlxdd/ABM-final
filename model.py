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
        # Calculate support from neighbors
        for neighbor in self.model.G.neighbors(self.unique_id):
            neighbor_agent = self.model.G.nodes[neighbor]['agent']
            if isinstance(neighbor_agent, SupportiveMember) and random.random() < 0.5:
                support_type = neighbor_agent.member_type
                support_value = SUPPORT_STRENGTHS[support_type]* random.uniform(0.5, 1.5)
                net_support += support_value
                self.support_network[support_type] += 1

        new_depression_index = self.depression_level.value - (net_support*self.model.support_effectiveness)
        print(new_depression_index)


        if new_depression_index < 0.5:
            self.depression_level = State.MILD
        elif new_depression_index < 1.5:
            self.depression_level = State.MODERATE
        else:
            self.depression_level = State.SEVERE


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


class DepressionSupportModel(Model):
    def __init__(self,num_agents=10, num_supportive=20, stigma_level=0.5, support_effectiveness=0.5):
        super().__init__()
        self.num_agents = num_agents
        self.num_supportive = num_supportive
        self.stigma_level = stigma_level
        self.support_effectiveness = support_effectiveness
        self.schedule = RandomActivation(self)
        self.G = nx.Graph()
        self.setup_agents()
        self.datacollector = DataCollector(
            model_reporters={
                "Average Depression": compute_average_depression
            }
        )

    def setup_agents(self):
        all_agents = []
        for i in range(self.num_agents):
            depression_level = State(random.choice([0, 1, 2]))
            resilience_factor = random.random()
            agent = Individual(i, self, depression_level, resilience_factor)
            self.schedule.add(agent)
            all_agents.append(agent)
            self.G.add_node(agent.unique_id, agent=agent)

        for j in range(self.num_agents, self.num_agents + self.num_supportive):
            member_type = random.choice(['therapist', 'family', 'friend'])
            supportive_member = SupportiveMember(j, self, member_type)
            self.schedule.add(supportive_member)
            self.G.add_node(supportive_member.unique_id, agent=supportive_member)

        for individual in [agent for agent in all_agents if isinstance(agent, Individual)]:
            supportive_members = [member for member in self.schedule.agents if isinstance(member, SupportiveMember)]

            for supportive_member in supportive_members:
                if random.random() < 0.2:
                    self.G.add_edge(individual.unique_id, supportive_member.unique_id)


def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
