from concurrent.futures import ThreadPoolExecutor
from neo4j import GraphDatabase
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from typing import List, Tuple, Dict, Any
import traceback
import os
import json


class graph_retrieval:
    def __init__(self, chat_history=None):

        self.NEO4J_URI = os.getenv("NEO4J_GRAPH_URI")
        self.NEO4J_USER = os.getenv("NEO4J_GRAPH_USER")
        self.NEO4J_PASSWORD = os.getenv("NEO4J_GRAPH_PASSWORD")
        self.driver = GraphDatabase.driver(self.NEO4J_URI, auth=(self.NEO4J_USER, self.NEO4J_PASSWORD))
        self.queries = {
            "Company": """
            MATCH (c:Company {name: $search_term})
            OPTIONAL MATCH (c)-[r]->(connected_node)
            RETURN c, collect({relationship: type(r), connected_node: connected_node}) AS connections
            """,
            "Member": """
            MATCH (m:Member {name: $search_term})
            OPTIONAL MATCH (m)-[r]->(outgoing_node)
            OPTIONAL MATCH (incoming_node)-[r2:HAS_MEMBER]->(m)
            RETURN m, 
            collect(DISTINCT {relationship: type(r), connected_node: outgoing_node}) AS outgoing_connections, 
            collect(DISTINCT {relationship: type(r2),  connected_node: incoming_node}) AS incoming_connections
            """,
            "Project": """
            MATCH (p:Project {name: $search_term})
            OPTIONAL MATCH (p)-[r]->(connected_node)
            RETURN p, collect({relationship: type(r), connected_node: connected_node}) AS connections
            """
        }

    def execute_query(self, tx, query, search_term):
        result = tx.run(query, search_term=search_term)
        return [record for record in result]

    def process_search_term(self, driver, search_term):
        results = {}
        try:
            with driver.session() as session:
                for node_type, query in self.queries.items():
                    result = session.execute_read(self.execute_query, query, search_term)
                    results[node_type] = result

        except Exception as e:
            print(f"Error executing queries for {search_term}: {e}")
        return search_term, results

    def collect_company(self, search_term, result):
        final_company_data = []
        for record in result:
            company_data = {"name": "", "about": "","description":"", "directory_link": "", "contact": "", "funding_stage": "",
                            "industry_tags": [], "focus_area": [], "technology": []}
            node = record[0]
            connections = record['connections']
            company_data["name"] = node["name"]
            company_data["about"] = node["about"]
            company_data["description"]=node["description"]
            company_data["directory_link"] = node["directoryLink"]
            company_data["contact"] = node["contactMethod"]
            for connection in connections:
                connected_node = connection.get("connected_node", {})
                relationship = connection.get("relationship", "")

                if relationship == "HAS_FUNDING_STAGE" and connected_node:
                    funding_stage = connected_node.get("stage", "")
                    if funding_stage:
                        company_data["funding_stage"] = funding_stage

                if relationship == "HAS_FOCUS_AREA" and connected_node:
                    focus_area = connected_node.get("name", "")
                    if focus_area:
                        if isinstance(focus_area, list):
                            company_data["focus_area"].extend(focus_area)
                        else:
                            company_data["focus_area"].append(focus_area)

                if relationship == "USES_TECHNOLOGY" and connected_node:
                    technology = connected_node.get("name", "")
                    if technology:
                        if isinstance(technology, list):
                            company_data["technology"].extend(technology)
                        else:
                            company_data["technology"].append(technology)

                if relationship == "HAS_INDUSTRY_TAG" and connected_node:
                    industry_tag = connected_node.get("title", "")
                    if industry_tag:
                        if isinstance(industry_tag, list):
                            company_data["industry_tags"].extend(industry_tag)
                        else:
                            company_data["industry_tags"].append(industry_tag)

            final_company_data.append(company_data)
            return final_company_data

    def collect_projects(self, search_term, result):
        final_projects_data = []
        #print("super------------->", result)
        for record in result:
            projects_data = {"name": "", "Description":"","directory_link": "", "contact_email": "", "tagline": ""}
            node = record[0]
            connections = record['connections']
            projects_data["name"] = node["name"]
            projects_data["directory_link"] = node["directoryLink"]
            projects_data["Description"]=node["Description"]
            #print("uuuuuuu-------------->", projects_data)

            for connection in connections:
                connected_node = connection.get("connected_node", {})
                relationship = connection.get("relationship", "")

                if relationship == "HAS_TAGLINE" and connected_node:
                    tagline = connected_node.get("text", "")
                    if tagline:
                        projects_data["tagline"] = tagline

                if relationship == "HAS_DESCRIPTION" and connected_node:
                    description = connected_node.get("text", "")
                    if description:
                        projects_data["Description"] = description


                if relationship == "HAS_CONTACT_EMAIL" and connected_node:
                    contact_email = connected_node.get("email", "")
                    if contact_email:
                        if isinstance(contact_email, list):
                            projects_data["contact_email"] = contact_email
                        else:
                            projects_data["contact_email"] = [contact_email]

            final_projects_data.append(projects_data)

        return final_projects_data

    def collect_members(self, search_term, result):
        final_members_data = []
        for record in result:
            members_data = {"name": "", "organization": "", "directory_link": "", "role": [], "skills":[], "office_hours": "","twitterHandler":"","email":""}
            node = record[0]
            connections = record['outgoing_connections']
            incoming_connections = record['incoming_connections']
            members_data["name"] = node["name"]
            members_data["twitterHandler"]=node['']
            members_data["organization"] = []

            for connection in incoming_connections:
                if connection['relationship'] == "HAS_MEMBER":
                    organization_node = connection.get("connected_node", {})
                    organization = organization_node.get('name') if organization_node else ""
                    if organization:
                        members_data["organization"] = organization

            # Handling connections
            for connection in connections:
                connected_node = connection.get("connected_node", {})

                if connection['relationship'] == "HAS_DIRECTORY_LINK":
                    directory_link = list(connected_node.items())[0][1] if connected_node else ""
                    if directory_link:
                        members_data["directory_link"] = directory_link

                if connection['relationship'] == "HAS_ROLE":
                    role = list(connected_node.items())[0][1] if connected_node else ""
                    if role:
                        members_data["role"].append(role)

                if connection['relationship'] == "HAS_SKILLS":
                    skills = list(connected_node.items())[0][1] if connected_node else ""
                    if skills:
                        members_data["skills"].append(skills)

                if connection['relationship'] == "HAS_OFFICE_HOURS":
                    office_hours = list(connected_node.items())[0][1] if connected_node else ""
                    if office_hours == "Not Available":
                        members_data["office_hours"] = False
                    else:
                        members_data["office_hours"] = True

            # Ensure default values for any keys that might not have been set
            if not members_data.get("organization"):
                members_data["organization"] = ""
            if not members_data.get("directory_link"):
                members_data["directory_link"] = ""
            if not members_data.get("role"):
                members_data["role"] = []
            if not members_data.get("office_hours"):
                members_data["office_hours"] = False

            final_members_data.append(members_data)
            #print("Members data-------->", members_data)
        return final_members_data

    def collect_data(self, search_term, results):
        final_dict = {"project": [], "company": [], "members": []}

        for node_type, result in results.items():
            if result:
                data = None
                if node_type == "Member":
                    data = self.collect_members(search_term, result)
                    final_dict["members"] += data
                if node_type == "Company":
                    data = self.collect_company(search_term, result)
                    final_dict["company"] += data
                if node_type == "Project":
                    data = self.collect_projects(search_term, result)
                    #print("data------------>", data)
                    final_dict["project"] += data
            else:
                continue

        return final_dict

    def print_results(self, search_term, results):
        for node_type, result in results.items():
            if result:
              #  print(f"Results for {node_type} with search term '{search_term}':")
                for record in result:
                    if node_type == "Member":
                        node = record['m']
                        outgoing_connections = record['outgoing_connections']
                        incoming_connections = record['incoming_connections']

                        #print("Node Properties:")
                        for key, value in node.items():
                           print(f"{key}: {value}")

                       # print("Outgoing Connections:")
                        for connection in outgoing_connections:
                            relationship_type = connection['relationship']
                            connected_node = connection['connected_node']

                            if connected_node:
                               # print(f"Relationship: {relationship_type}")
                               # print("Connected Node Properties:")
                                for key, value in connected_node.items():
                                   print(f"{key}: {value}")
                            else:
                                print(f"Relationship: {relationship_type}, Connected Node: None")

                       # print("Incoming Connections:")
                        for connection in incoming_connections:
                            relationship_type = connection['relationship']
                            connected_node = connection['connected_node']
                            for key, value in connected_node.items():
                                if key == "name":
                                    key = "Company_Name"
                              #  print(f"{key}: {value}")
                            else:
                                print(f"Relationship: {relationship_type}, Connected Node: None")

                       # print()
                    else:
                        node = record[0]
                        connections = record['connections']

                        print("Node Properties:")
                        for key, value in node.items():
                            print(f"{key}: {value}")

                     #   print("Connected Nodes and Relationships:")
                        for connection in connections:
                            relationship_type = connection['relationship']
                            connected_node = connection['connected_node']
                        #    print(f"Relationship: {relationship_type}")
                            connected_node_properties = ', '.join(f"{k}: {v}" for k, v in connected_node.items())
                         #   print(f"Connected Node Properties: {connected_node_properties}")
                       # print()
                else:
                    continue

    def execute_query(self, tx, query, search_term):
        result = tx.run(query, search_term=search_term)
        return [record for record in result]

    def process_search_term(self, driver, search_term):
        results = {}
        try:
            with driver.session() as session:
                for node_type, query in self.queries.items():
                    result = session.execute_read(self.execute_query, query, search_term)
                    results[node_type] = result

        except Exception as e:
            print(f"Error executing queries for {search_term}: {e}")
        return search_term, results


    def print_results(self, search_term, results):
        for node_type, result in results.items():
            if result:
          #      print(f"Results for {node_type} with search term '{search_term}':")
                for record in result:
                    if node_type == "Member":
                        node = record['m']
                        outgoing_connections = record['outgoing_connections']
                        incoming_connections = record['incoming_connections']

                    #    print("Node Properties:")
                        for key, value in node.items():
                           print(f"{key}: {value}")

                    #    print("Outgoing Connections:")
                        for connection in outgoing_connections:
                            relationship_type = connection['relationship']
                            connected_node = connection['connected_node']

                            if connected_node:
                             #   print(f"Relationship: {relationship_type}")
                              #  print("Connected Node Properties:")
                                for key, value in connected_node.items():
                                    print(f"{key}: {value}")
                            else:
                                print(f"Relationship: {relationship_type}, Connected Node: None")

                      #  print("Incoming Connections:")
                        for connection in incoming_connections:
                            relationship_type = connection['relationship']
                            connected_node = connection['connected_node']
                            for key, value in connected_node.items():
                                if key == "name":
                                    key = "Company_Name"
                             #   print(f"{key}: {value}")
                            else:
                                print(f"Relationship: {relationship_type}, Connected Node: None")

                     #   print()
                    else:
                        node = record[0]
                        connections = record['connections']

                       # print("Node Properties:")
                        for key, value in node.items():
                            print(f"{key}: {value}")

                       # print("Connected Nodes and Relationships:")
                        for connection in connections:
                            relationship_type = connection['relationship']
                            connected_node = connection['connected_node']
                         #   print(f"Relationship: {relationship_type}")
                            connected_node_properties = ', '.join(f"{k}: {v}" for k, v in connected_node.items())
                         #   print(f"Connected Node Properties: {connected_node_properties}")
                       # print()
                else:
                    continue

    def get_answer(self, search_terms):
        final_dict = {"project": [], "company": [], "members": []}
        try:
            search_terms = [term for term in search_terms]
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.process_search_term, self.driver, term) for term in search_terms]

                for future in futures:
                    search_term, results = future.result()
                    # self.print_results(search_term, results)
                    data = self.collect_data(search_term, results)
                    final_dict["members"].extend(data["members"])
                    final_dict["company"].extend(data["company"])
                    final_dict["project"].extend(data["project"])
        except Exception as e:
            print(f"Error in get_answer: {e}")

        return final_dict

    def extract_entities(self, query):
        openai_llm = ChatOpenAI(model="gpt-3.5-turbo")
        prompt_template = PromptTemplate(
            input_variables=[query],
            template="""
            Extract the names of people and organizations from the following text and categorize them as 'Person' or 'Organization':
            Text: "{query}"
            Output the results in a JSON format with 'person' and 'organization' lists.
            """
        )
        chain = LLMChain(llm=openai_llm, prompt=prompt_template)
        response = chain.run({"query": query})
        entities = json.loads(response)
        return entities

