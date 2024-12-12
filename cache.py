import redis
import json
from langchain_core.messages import AIMessage, HumanMessage
import json
import os
import ast
import json
import asyncio
from dotenv import load_dotenv
load_dotenv()

def serialize(obj):
    return json.dumps(obj.__dict__)

def deserialize(json_str, cls):
    return cls(**json.loads(json_str))



class cache():
    client = None
    base_chat_history = []
    host = os.getenv('host')
    port= os.getenv('port')
    password =os.getenv('password')
    time_to_expire_s = os.getenv('cache_time_to_expire_s')
    def get_cache_client(self):
        return self.clientS
    async def append_chat_history(self,uid,question, ai_msg):
        #chat_history.extend([HumanMessage(content=question), AIMessage(content = ai_msg)])
        history_size = 5
        print(f'type of chat history {type(self.base_chat_history)}')
        self.base_chat_history.extend([question,ai_msg])
        if len(self.base_chat_history)>history_size * 5:
            self.base_chat_history.pop(0)
            self.base_chat_history.pop(0)
        #store in redis
        #print(type(chat_history))
        # print("chat_history")
        # print(self.base_chat_history)
        self.base_chat_history = {"chat_history":str(self.base_chat_history)}
        print(self.base_chat_history)
        #chat_history = serialize(chat_history)
        #json_data = json.dumps(chat_history, default=lambda o: o.__dict__)
        #chat_history = vars(chat_history)
        # Store objects in Redis
        
        redis_resp = self.client.hmset(uid, self.base_chat_history)
        self.client.expire(uid, self.time_to_expire_s)
        return redis_resp


    def flush_chat_history(self,uid):
        #json_data = json.dumps([])
        return self.client.delete(uid)
    def process_chat_history(self,history):
        processed_history = []
        for i in range(len(history)):
            if i%2 ==0 :
                processed_history = processed_history +[HumanMessage(content=str(history[i]))]
            else:
                processed_history = processed_history +[AIMessage(content = str(history[i]))]
        return processed_history

    async def retrieve_chat_history(self,uid):
        #history = self.client.get(uid)
        history = self.client.hgetall(uid)
        # print("retrieved chat history")
        # print(history)
        if len(history.keys())>0 :
            #history = deserialize(history, list())
            history = history[b"chat_history"]
            history = history.decode('utf-8')
            history = ast.literal_eval(history)
            # print(history)
            # print(type(history))
            self.base_chat_history = history
            processed_chat_history = self.process_chat_history(history)

            #print(history)
            return history
        else:
            self.base_chat_history = []
            return []

    def connect_cache(self):
        client = redis.Redis(host = self.host, port=self.port,password = self.password)
        client.ping()
        return client
    
    def __init__(self) -> None:
        self.client = self.connect_cache()