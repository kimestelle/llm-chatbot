import torch
from process_data import get_pad_id

def load_tokenized_data(file_path):
    with open(file_path, 'r') as f:
        tokenized_data = []
        for line in f:
            token_ids = list(map(int, line.strip().split()))
            tokenized_data.append(token_ids)
    return tokenized_data

def pad_and_save_data(tokenized_data, output_file_path, pad_id):
    max_seq_len = max(len(tokens) for tokens in tokenized_data)

    padded_sentences = [tokens + [pad_id] * (max_seq_len - len(tokens)) for tokens in tokenized_data]

    input_tensor = torch.tensor(padded_sentences, dtype=torch.long, device='cpu')  # Adjust device as needed

    padded_sentences_list = input_tensor.tolist()

    with open(output_file_path, 'w') as f:
        for sentence in padded_sentences_list:
            f.write(' '.join(map(str, sentence)) + '\n')

    print(f"Padded tokenized data has been written to '{output_file_path}'.")

if __name__ == "__main__":
    tokenized_data = load_tokenized_data('processed_plato_as_ids.txt')

    pad_id = get_pad_id()

    pad_and_save_data(tokenized_data, 'padded_processed_plato.txt', pad_id)
