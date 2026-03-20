import asyncio
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams


async def main():

    engine_args = AsyncEngineArgs(
        model="lmsys/vicuna-13b-v1.5",
        tensor_parallel_size=4,
        dtype="float16",
        max_model_len=2048,              # temporarily lower
        gpu_memory_utilization=0.8,     # lower from 0.9
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)

    prompt = "Explain tensor parallelism in LLMs."

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=128
    )

    request_id = "req_1"

    async for output in engine.generate(
        prompt,
        sampling_params,
        request_id
    ):
        if output.finished:
            print(output.outputs[0].text)


asyncio.run(main())