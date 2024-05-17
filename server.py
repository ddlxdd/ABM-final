import mesa
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import NetworkModule, ChartModule
from mesa.visualization.UserParam import Slider
from model import DepressionSupportModel, Individual, State, SupportiveMember

def network_portrayal(G):
    def node_color(agent):
        if hasattr(agent, 'depression_level'):
            colors = {State.MILD: "#ADD8E6", State.MODERATE: "#FFA500", State.SEVERE: "#FF0000"}
            return colors.get(agent.depression_level, "#808080")
        elif isinstance(agent, SupportiveMember):
            type_colors = {'therapist': "#FFD700", 'family': "#32CD32", 'friend': "#00BFFF"} 
            return type_colors.get(agent.member_type, "#808080")
        return "#808080"

    def node_shape(agent):
        if isinstance(agent, SupportiveMember):
            return "square"
        return "circle"

    portrayal = {
        "nodes": [{
            "size": 10 if hasattr(agent, 'depression_level') else 8,
            "color": node_color(agent),
            "shape": node_shape(agent),
            "tooltip": f"id: {agent.unique_id}<br>type: {'Individual' if hasattr(agent, 'depression_level') else f'Supportive Member ({agent.member_type})'}<br>state: {getattr(agent, 'depression_level', 'N/A').name if hasattr(agent, 'depression_level') else 'N/A'}",
        } for (_, agent) in G.nodes.data("agent")],
        "edges": [{
            "source": source,
            "target": target,
            "color": "#e8e8e8",
            "width": 2
        } for (source, target) in G.edges]
    }

    return portrayal

network = NetworkModule(network_portrayal, 500, 500)

chart = ChartModule([
    {"Label": "Average Depression", "Color": "Black"}
], data_collector_name='datacollector')

model_params = {
    "num_agents": Slider("Number of Agents", 10, 2, 50, 1),
    "num_supportive": Slider("Number of Supportive Members", 20, 1, 50, 1),
    "stigma_level": Slider("Stigma Level", 0.5, 0.0, 1.0, 0.05),
    "support_effectiveness": Slider("Support Effectiveness", 0.5, 0.0, 1.0, 0.05)
}

server = ModularServer(DepressionSupportModel,
                       [network, chart],
                       "Depression and Support Networks Model",
                       model_params)
server.port = 8521
