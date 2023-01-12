from model import CodeBert_Seq2Seq

# 初始化模型
model = CodeBert_Seq2Seq(codebert_path = 'microsoft/codebert-base', decoder_layers = 6, fix_encoder = False, beam_size = 10,
                         max_source_length = 300, max_target_length = 32, load_model_path =None)

# # 加载微调过的模型
# model = CodeBert_Seq2Seq(codebert_path='microsoft/codebert-base', decoder_layers=6, fix_encoder=False, beam_size=10,
#                          max_source_length=300, max_target_length=32,
#                          load_model_path='valid_output_NoEdit/checkpoint-last/pytorch_model.bin')

# 模型训练
model.train(train_filename='data/train_ast.json', train_batch_size=12, num_train_epochs=5, learning_rate=5e-5,
            do_eval=True, dev_filename='data/valid_ast.json', eval_batch_size=16, output_dir='valid_output')

# 模型测试
model.test(test_filename='data/test_ast.json', test_batch_size=10, output_dir='test_output')


