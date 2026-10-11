import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    将文件划分为可以独立计数的片段。
    如果边界重合，返回的片段数可能少于期望值。
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # 获取文件总大小，单位为字节。
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # 均匀选取片段边界的初始候选位置。
    # 每个片段从上一个边界开始，不包含下一个边界。
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # 每次向前读取 4096 字节。

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # 从候选边界位置开始读取。
        while True:
            mini_chunk = file.read(mini_chunk_size)  # 读取一小段数据。

            # 如果到达文件末尾，则将该边界设为文件末尾。
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 在当前数据片段中查找特殊词元。
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # 去除重复边界；最终边界数可能少于期望的片段数。
    return sorted(set(chunk_boundaries))


## 用法
with open(..., "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # 以下是串行示例；也可以将各组 start/end 边界分发给多个进程，并行处理。
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # 对当前片段进行预分词（Pre-tokenization），并记录各预词元的出现次数。
