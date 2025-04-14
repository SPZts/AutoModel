import asyncio

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent, SocietyOfMindAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


def model_03_init(ds_model_client) -> SocietyOfMindAgent:
    model_03_selector = AssistantAgent(
        name="Model_03_Selector",
        model_client=ds_model_client,
        description="指标03的模型选择器",
        system_message="你负责理解用户的需求，并思考哪个模型符合用户的需求。若有，返回模型名称；若无，返回“skip”"
                       "下面是模型名称以及对应的模型说明"
                       "NODISASTERPLANNING：化工园区有居民居住，且未提供搬迁方案和保障措施"
    )
    model_03_coder = AssistantAgent(
        name="Model_03_Coder",
        model_client=ds_model_client,
        description="指标03的代码器",
        system_message="你根据Model_03_Selector回答的模型名称与园区id，生成对应的代码。示例代码如下，请把#{modelName}变成Model_03_Selector回答的内容"
                       '''
                       import requests

                        def call_evaluate_api(model: str, park_id: str) -> bool:
                        url = "http://2.93.4.30:8765/evaluate"
                        params = {
                            "model": #{modelName}
                        }

                        try:
                            response = requests.get(url, params=params)
                            response.raise_for_status()

                            text = response.text.strip().lower()
                            if text == "true":
                                return True
                            elif text == "false":
                                return False
                            else:
                                print(f"返回值无法识别: {text}")
                                return False
                        except requests.RequestException as e:
                            print(f"请求失败: {e}")
                            return False
                       '''
    )
    # code_executor = DockerCommandLineCodeExecutor(work_dir="coding")
    # model_03_executor = CodeExecutorAgent(
    #     name="Model_03_Executor",
    #     code_executor=code_executor,
    #     description="指标03的代码执行器",
    # )
    model_03_summary = AssistantAgent(
        name="Model_03_Summary",
        model_client=ds_model_client,
        description="指标03的总结代理",
        system_message="你负责理总结Model_03_Executor的执行结果,如果它没有回答,默认结果为True"
    )
    model_03_team = RoundRobinGroupChat(
        participants=[model_03_selector,
                      model_03_coder,
                      # model_03_executor,
                      model_03_summary],
        # max_turns=4,
        max_turns=3,
        termination_condition=TextMentionTermination("skip")
    )
    model_03_agent = SocietyOfMindAgent(
        name="Model_03_Agent",
        team=model_03_team,
        model_client=ds_model_client,
        description="指标03的team",
        response_prompt="精简Model_03_Summary的发言为一句话，以陈述句说出"
    )

    return model_03_agent
