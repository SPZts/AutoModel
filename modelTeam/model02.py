import asyncio

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent, SocietyOfMindAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


def model_02_init(ds_model_client) -> SocietyOfMindAgent:
    model_02_selector = AssistantAgent(
        name="Model_02_Selector",
        model_client=ds_model_client,
        description="指标02的模型选择器",
        system_message="你负责理解用户的需求，并思考哪个模型符合用户的需求。若有，返回模型名称；若无，返回“skip”"
                       "下面是模型名称以及对应的模型说明"
                       "NOAPPROVALDOCUMENT：无《四至范围的批复》文件"
                       "SCOPEACROSSCOMPANY：存在园区四至范围穿越企业的情况"
                       "AREADIFFERENT：规划范围和规划面积与《四至范围的批复》文件中不一致"
                       "下面是只需要一个模型的回复示例:NOAPPROVALDOCUMENT;如果需要多个模型,用英文逗号分开,比如:NOAPPROVALDOCUMENT,AREADIFFERENT"
    )
    model_02_coder = AssistantAgent(
        name="Model_02_Coder",
        model_client=ds_model_client,
        description="指标02的代码器",
        system_message="你根据Model_02_Selector回答的模型名称与园区id，生成对应的代码。示例代码如下，请把#{modelName}变成Model_02_Selector回答的内容"
                       "如果Model_02_Selector返回了多个模型，那么你的代码里应该调用多次requests"
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
    # model_02_executor = CodeExecutorAgent(
    #     name="Model_02_Executor",
    #     code_executor=code_executor,
    #     description="指标02的代码执行器",
    # )
    model_02_summary = AssistantAgent(
        name="Model_02_Summary",
        model_client=ds_model_client,
        description="指标02的总结代理",
        system_message="你负责理总结Model_02_Executor的执行结果,如果它没有回答,默认结果为True"
    )
    model_02_team = RoundRobinGroupChat(
        participants=[model_02_selector,
                      model_02_coder,
                      # model_02_executor,
                      model_02_summary],
        # max_turns=4,
        max_turns=3,
        termination_condition=TextMentionTermination("skip")
    )
    model_02_agent = SocietyOfMindAgent(
        name="Model_02_Agent",
        team=model_02_team,
        model_client=ds_model_client,
        description="指标02的team",
        response_prompt="精简Model_02_Summary的发言为一句话，以陈述句说出"
    )

    return model_02_agent
