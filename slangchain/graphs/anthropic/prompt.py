"""AgentSupervisor prompts"""
NEXT = "next"
FINISH = "FINISH"
SUPERVISOR = "supervisor"
SUPERVISOR_SYSTEM_PROMPT = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status."
    " Be very economical and use the minimum workers to achieve your task."
    " When finished, respond with FINISH."
    "Given the conversation below, who should act next?"
    " Or should we FINISH? Select one of: {options}"
)
