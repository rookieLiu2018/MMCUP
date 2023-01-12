from tokenizers import ByteLevelBPETokenizer
import os

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=["data/train_ast.json","data/valid_ast.json","data/test_ast.json"],
                vocab_size=50265, min_frequency=1, special_tokens=["<s>", "<pad>", "</s>", "<unk>"])
if os.path.exists("./tokenize") == False:
    os.makedirs("./tokenize")
tokenizer.save_model("./tokenize")