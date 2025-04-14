import asyncio
from typing import Sequence

from autogen_agentchat.agents import AssistantAgent, SocietyOfMindAgent, CodeExecutorAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, AgentEvent, ChatMessage
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient

import modelTeam.model01
import modelTeam.model02
import modelTeam.model03


# Define a model client. You can use other model client that implements
# the `ChatCompletionClient` interface.
ds_model_client = OpenAIChatCompletionClient(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key="sk-0833c4be10814e759346c7ae6305de0b",
    model_info={
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "family": "unknown",
    },
)
dsr1_model_client = OpenAIChatCompletionClient(
    model="deepseek-chat-r1",
    base_url="https://api.deepseek.com",
    api_key="sk-0833c4be10814e759346c7ae6305de0b",
    model_info={
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "family": "unknown",
    },
)
model_list = []

async def main() -> None:
    indicator_selector = AssistantAgent(
        name="Indicator_Selector",
        model_client=ds_model_client,
        system_message="你负责理解用户的需求,用户的需求需要一项或多项指标完成,你要想出哪些指标可以完成用户的任务,仅回复指标名称.以下是现有的指标"
                       "Model_01_Agent：手续流程齐全;"
                       "Model_02_Agent：‘四至’范围合理;"
                       "Model_03_Agent：园区内不允许居民居住;"
                       "下面是只需要一个指标的回复示例:Model_01_Agent;如果需要多个指标,用英文逗号分开,比如:Model_01_Agent,Model_03_Agent"
    )

    summary_agent = AssistantAgent(
        name="SummaryAgent",
        model_client=ds_model_client,
        system_message="你负责总结各个ModelSummaryAgent的发言,并把他们整合成一段话输出,在输出的最后加上APPROVE。特别注意，整理发言控制在50字以内"
    )

    # 初始化各个模型的Agent
    model_01_agent = modelTeam.model01.model_01_init(ds_model_client)
    model_02_agent = modelTeam.model02.model_02_init(ds_model_client)
    model_03_agent = modelTeam.model03.model_03_init(ds_model_client)

    # 定义终止条件  如果提到APPROVE则终止对话
    text_termination = TextMentionTermination("APPROVE")
    # 定义终止条件，在20条信息后停止任务
    max_message_termination = MaxMessageTermination(20)
    # 使用`|` 运算符组合终止条件，在满足任一条件时停止任务
    termination = text_termination | max_message_termination

    # 定义agent选择器
    def selector_func(messages: Sequence[AgentEvent | ChatMessage]) -> str | None:
        global model_list

        if len(model_list) != 0:
            del model_list[0]
            if len(model_list) != 0:
                return model_list[0]
            elif len(model_list) == 0:
                return "SummaryAgent"

        if messages[-1].source == "user":
            return "Indicator_Selector"

        if messages[-1].source == "Indicator_Selector":
            model_list = messages[-1].content.split(",")
            return model_list[0]

        return None

    # 定义Team Team的类型选择为SelectorGroupChat
    main_team = SelectorGroupChat(
        participants=[indicator_selector, summary_agent, model_01_agent, model_02_agent, model_03_agent],
        termination_condition=termination,
        max_turns=None,
        model_client=ds_model_client,
        selector_func=selector_func
    )

    # 1、运行team并使用官方提供的Console工具以适当的格式输出
    # stream = main_team.run_stream(task="我要看看南京江北化工园区手续是否齐全")
    # await Console(stream)
    await Console(main_team.run_stream(task="我要看看南京江北化工园区有没有居民居住"))

    # # 2、恢复任务 同时保留上一个任务的上下文
    # stream = main_team.run_stream(task="将这首诗用英文写一遍。")
    # await Console(stream)

    # # 3、恢复前一个任务 不用传递具体任务 team将从上个任务中断的地方继续支持
    # stream = main_team.run_stream()
    # await Console(stream)


if __name__ == '__main__':
    # 运行main
    asyncio.run(main())
