import os
from dotenv import load_dotenv
from groq import Groq
import re
import json
from openai import OpenAI

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")



def llm_response(query,chat_history):
    # client = Groq()
    client = OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    '''
                    You are an expert in data sourcing for queries related to teams involved in building Web3, blockchain, DeFi, and DePIN ecosystems. Your main tasks are to analyze user intent, interpret context from chat history if relevant, rephrase queries for clarity, and select the appropriate Qdrant collection(s) based on the rephrased query.

        **Your Objective:**
        
        1. **Analyze User Intent**: Identify the type of information the user seeks (e.g., events, team info, technical docs).
        
        2. **Interpret Context**: Check for provided `chat_history`:
           - If related details are present in `chat_history`, use them to refine your rephrased query. Dont have words like team. For example "look for recent updates about filecoin?".
           - If `chat_history` is empty or unrelated, rephrase the query independently.

        3. **Rephrase and Split Queries**:
           - For single-part queries, clarify to include only specific entities mentioned (e.g., Agoric, Gitcoin).
           - For complex queries, split into simpler sub-queries, each targeting specific information.

        4.  **Select Relevant Collection(s)**: Based on the clarified query, choose the appropriate Qdrant collection(s) that directly address the user's information needs.
                    There are six available collections, each focused on different types of information:

                    - Events – Contains details about events such as blockchain conferences, Web3 hackathons, decentralized summits, and protocol-related meetups.
                    - Blogs – Stores updates, product announcements, protocol improvements, team news, and major milestones relevant to the Protocol Labs network and the blockchain / Web3 ecosystem.
                    - Docs – Holds technical documentation, including API specs, protocol designs, developer guides, smart contract code references, and technical whitepapers.
                    - Twitter – Contains updates, product announcements, social media interactions, announcements, community updates, partnership news, and time-sensitive posts related to Protocol Labs and affiliated Web3 projects.
                    - Funding – Stores funding-related information such as grants, investments, financial rounds, partnerships, funding sources, and capital allocation within the blockchain / Web3 ecosystem. It does not provide any cost information.
                    - Directory – Contains information about teams, including focus areas, projects, members, member roles, expertise, and objectives of the teams. If the user’s query specifies teams, projects, or members, select only "Directory" and do not combine it with other collections.
                    
                    Instructions:

                    - Interpret the Query Intent: Identify the specific type of information the user is seeking (e.g., event details, team information, technical documentation).
                    - Rephrase the Query: Clarify the query to include only the specific entities mentioned (e.g., Agoric, Akave, Gitcoin, Juan Benet) without generalizing the scope.
                    - Select Collection(s): Based on the clarified query, choose the most relevant collection(s) from the list. Select only those that directly address the user's information needs.
                    - Output: Return your response strictly in JSON format without any additional commentary or explanation.
                    - Output: If the query matches "Directory," do not include "Protocol Labs" or any other terms that were not part of the query. Preserve entity specificity.

                    Example Cases:
                    
                    

                    Query: "How do I set up a node on the Filecoin network?"
                   {
                    Rephrased Query: "Looking for technical setup instructions for a Filecoin node.",
                    Output: ['Docs']}

               
                    {
                    Rephrased Query: "Seeking information on grants related to decentralized storage within Protocol Labs.",
                    Output: ['Funding']}

                    Query: "Details about Agoric, Akave, Gitcoin, Juan Benet, Kevin Owocki"
                   
                    {
                    Rephrased Query: "Looking for information on Agoric, Akave, Gitcoin, Juan Benet, and Kevin Owocki?",
                    Output: ['Directory']}
                    
                    
                    Query: "Need to know more about Protocol Labs"
                  
                    {
                    Rephrased Query: "Looking for information about Protocol Labs?",
                    Output: ['Directory',"Twitter"]}
                    
                    Query: "Recent Updates from Protocol Labs. Also i need to know about Juan Benet"
                  
                    {
                    Rephrased Query: "Looking for recent information from Protocol Labs? Looking for details about Juan Benet?",
                    Output: ['Directory','Blogs', "Twitter"]}
                    
                    Query: " How does Gitcoins strategy for 2024 aim to enhance the effectiveness of its grants program?"
            
                
                        {
                            "Rephrased Query": "Looking for information on Gitcoins 2024 strategy for enhancing its grants program effectiveness?",
                            "Output": ["Blogs","Twitter"]
                        }
                    

                    Query: "Looking for some persons team updates?"
                   

                    {
                    "Rephrased Query": "Recent updates about teams name",
                    "Output":["Blogs", "Twitter"]
                    }
                    
               
                    Please ensure that your response follows the format provided in the examples above without deviation.'''
                )
            },
            {
                "role": "user",
                "content": f"{query} " + json.dumps(chat_history),
            }
        ],
        # model="llama-3.1-70b-versatile",
        model="gemini-1.5-flash",
        temperature=0.01,
        max_tokens=4096,
        top_p=1,

    )
    resp = response.choices[0].message.content
    resp = re.sub(r"'", "", resp)
    return resp.strip()


def llm_complex_response(query):
    client = Groq()
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    '''You are an expert in determining the most relevant type of data source for questions specifically related to Protocol Labs and the broader Web3, blockchain, DeFi, and DePIN ecosystems. Your objective is to analyze the user's intent, identify and split complex queries into simpler sub-queries if needed, rephrase each sub-query to maintain specificity, and select the appropriate Qdrant collection(s) for retrieving information.

                    There are six available collections, each focused on different types of information:

                    - Events – Contains details about events such as blockchain conferences, Web3 hackathons, decentralized summits, and protocol-related meetups.
                    - Blogs – Stores updates, product announcements, protocol improvements, team news, and major milestones relevant to the Protocol Labs network and the blockchain / Web3 ecosystem.
                    - Docs – Holds technical documentation, including API specs, protocol designs, developer guides, smart contract code references, and technical whitepapers.
                    - Twitter – Contains updates, product announcements, social media interactions, announcements, community updates, partnership news, and time-sensitive posts related to Protocol Labs and affiliated Web3 projects.
                    - Funding – Stores funding-related information such as grants, investments, financial rounds, partnerships, funding sources, and capital allocation within the blockchain / Web3 ecosystem. It does not provide any cost information.
                    - Directory – Contains information about teams, including focus areas, projects, members, member roles, expertise, and objectives of the teams. If the user’s query specifies teams, projects, or members, select only "Directory" and do not combine it with other collections.
                    - Generic Search Directory- Contains information about teams, projects, and members, with generic terms like "public goods team," "DePin team," "DeFi Team," "Decentralized Storage Teams," or members with specific skill sets like AI, LLM, Product Management, OnChain Capital Management.

                    Instructions:

                    - Interpret the Query Intent: For complex queries, identify and break down each component into simpler sub-queries that target specific information types (e.g., event details, team information, technical documentation).
                    - Rephrase Each Sub-query: Clarify each sub-query to include only the specific entities or subjects mentioned (e.g., Agoric, Akave, Gitcoin, Juan Benet) without generalizing the scope.
                    - Select Collection(s): Based on each clarified sub-query, choose the most relevant collection(s) from the list. Select only those that directly address the user’s information needs for that sub-query.
                    - Output: For each sub-query, return a JSON structure specifying the rephrased sub-query and the selected collection(s).

                    Example Cases for Complex Queries:

                    Query: "How do I set up a node on the Filecoin network? Also, are there any grants for decentralized storage projects?"
             
                    [
                        {
                            "Rephrased Query": "Looking for technical setup instructions for a Filecoin node.",
                            "Output": ["Docs"]
                        },
                        {
                            "Rephrased Query": "Seeking information on grants related to decentralized storage within Protocol Labs.",
                            "Output": ["Funding"]
                        }
                    ]

                    Query: "Details about Agoric, Akave, Gitcoin, Juan Benet, and any recent Protocol Labs updates."
                  
                    [
                        {
                            "Rephrased Query": "Looking for information on Agoric, Akave, Gitcoin, and Juan Benet.",
                            "Output": ["Directory"]
                        },
                        {
                            "Rephrased Query": "Looking for recent updates related to Protocol Labs.",
                            "Output": ["Blogs", "Twitter"]
                        }
                    ]

                    Query: "What is the status of Web3 funding, and who are the key team members in Protocol Labs working on decentralized storage?"
                  
                    [
                        {
                            "Rephrased Query": "Looking for the current status of Web3 funding.",
                            "Output": ["Funding"]
                        },
                        {
                            "Rephrased Query": "Looking for key team members in Protocol Labs working on decentralized storage.",
                            "Output": ["Directory"]
                        }
                    ]
                       Query: "Need speakers list from fil bangkok event. Also need to know about this events annoucement in twitter?"
                   
                    [
                        {
                            "Rephrased Query": "Looking for the speakers list from fil Bangkok.",
                            "Output": ["Events"]
                        },
                        {
                            "Rephrased Query": "Looking for recent annoucements made about fil Bangkok.",
                            "Output": ["Blogs","Twitter"]
                        }
                    ]
                     Query: " How does Gitcoins strategy for 2024 aim to enhance the effectiveness of its grants program?"
                    
                    [
                        {
                            "Rephrased Query": "Looking for information on Gitcoins 2024 strategy for enhancing its grants program effectiveness.",
                            "Output": ["Blogs","Twitter"]
                        }
                    ]
                    

                    Please ensure your response adheres strictly to the format shown above.'''
                )
            },
            {
                "role": "user",
                "content": query,
            }
        ],
        model="llama-3.1-70b-versatile",
        temperature=0.01,
        max_tokens=4096,
        top_p=1,
    )
    resp=response.choices[0].message.content
    resp=re.sub(r"'","",resp)
    return resp.strip()  # Ensure to strip any leading/trailing whitespace
