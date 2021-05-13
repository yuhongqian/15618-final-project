from transformers import BigBirdTokenizer, BigBirdForSequenceClassification, BigBirdConfig, PyTorchBenchmark, PyTorchBenchmarkArguments
from itertools import product
import torch
import torch.utils.benchmark as benchmark
import csv
import time
import traceback
torch.backends.cudnn.benchmark = True

input_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore " \
             "et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi " \
             "ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse " \
             "cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa" \
             " qui officia deserunt mollit anim id est laborum."


def time_single_forward(model, inputs):
    # print(inputs)
    input_ids = inputs["input_ids"].to(torch.device("cuda"))
    attention_mask = inputs["attention_mask"].to(torch.device("cuda"))
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs


def time_foward_backward(model, inputs):
    input_ids = inputs["input_ids"].to(torch.device("cuda"))
    attention_mask = inputs["attention_mask"].to(torch.device("cuda"))

    torch.cuda.synchronize()
    forward_start = time.time()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    torch.cuda.synchronize()
    forward_end = time.time()

    torch.cuda.synchronize()
    backward_start = time.time()
    torch.sum(outputs.logits).backward()
    torch.cuda.synchronize()
    backward_end = time.time()

    return (forward_end - forward_start) * 1000, (backward_end - backward_start) * 1000


def pytorch_benchmark(batch_sizes, sequence_lengths, nums_random_blocks, output_path, attention_type="block_sparse"):
    # Compare takes a list of measurements which we'll save in results.
    device = torch.device("cuda")

    fp = open(output_path, "w")
    writer = csv.writer(fp)
    writer.writerow(["batch_size", "seq_length", "r", "forward time (ms)", "bakward time (ms)"])
    tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
    for b, n, r in product(batch_sizes, sequence_lengths, nums_random_blocks):
        print(b, n, r)
        inputs = tokenizer([input_text for _ in range(b)], max_length=n, truncation=True, return_tensors="pt")
        config = BigBirdConfig.from_pretrained("google/bigbird-roberta-base", attention_type=attention_type)
        model = BigBirdForSequenceClassification.from_pretrained("google/bigbird-roberta-base", config=config)
        model.to(device)
        try:
            torch.cuda.synchronize()
            forward_time = 0
            backward_time = 0
            for _ in range(10):
                forward_elapse, backward_elapse = time_foward_backward(model, inputs)

                forward_time += forward_elapse
                backward_time += backward_elapse
            forward_time /= 10
            backward_time /= 10
            print(forward_time, backward_time)
            writer.writerow([b, n, r, forward_time, backward_time])
        except Exception as e:
            print("Error:", e)
            traceback.print_exc()

    fp.close()


def main():
    # batch_sizes = [2]
    # sequence_lengths = [256]
    # nums_random_blocks = [2]
    batch_sizes = [1, 2, 4, 8, 16, 32, 128, 256, 512]
    sequence_lengths = [32, 64, 128, 256, 512]
    nums_random_blocks = [2, 4, 8, 16, 32, 64, 128]
    pytorch_benchmark(batch_sizes, sequence_lengths, nums_random_blocks, output_path="full_benchmark.txt",
                      attention_type="original_full")
    # batch_sizes = [1, 2, 4, 8, 16, 32, 128, 256, 512]
    # sequence_lengths = [1024, 2048]
    # nums_random_blocks = [2, 4, 8, 16, 32, 64, 128]
    # pytorch_benchmark(batch_sizes, sequence_lengths, nums_random_blocks, output_path="sparse_benchmark.txt",
    #                   attention_type="block_sparse")


if __name__ == '__main__':
    main()

