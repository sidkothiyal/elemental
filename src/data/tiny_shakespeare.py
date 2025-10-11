import pathlib

import dill
import lz4.frame as lz4
import torch


def get_vocab(text: str) -> tuple[list[str], int]:
    """Get the vocabulary of the text."""
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print("vocab:", "".join(chars))
    print(f"vocab size: {vocab_size}")
    return chars, vocab_size


if __name__ == "__main__":
    data_dir = pathlib.Path("data/tiny_shakespeare/data.txt")

    with open(data_dir, "r", encoding="utf-8") as f:
        text = f.read()

    print(len(text))
    print(text[:1000])

    chars, vocab_size = get_vocab(text)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda string: [stoi[character] for character in string]
    decode = lambda letters: "".join([itos[letter.item()] for letter in letters])

    print(encode("hello there"))
    print(decode(encode("hello there")))

    data = torch.tensor(encode(text), dtype=torch.long)
    print(data.shape, data.dtype)
    print(data[:100])

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(train_data.shape, val_data.shape)

    # save the data
    with lz4.open("data/tiny_shakespeare/train_data.pkl.lz4", "wb") as f:
        dill.dump((chars, stoi, itos, train_data), f)

    with lz4.open("data/tiny_shakespeare/val_data.pkl.lz4", "wb") as f:
        dill.dump((chars, stoi, itos, val_data), f)
