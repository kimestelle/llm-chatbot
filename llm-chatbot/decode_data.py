import sentencepiece as spm

def load_token_ids(file_path):
    with open(file_path, 'r') as f:
        tokenized_data = []
        for line in f:
            token_ids = list(map(int, line.strip().split()))
            tokenized_data.append(token_ids)
    return tokenized_data

def decode_token_ids(tokenized_data, sp):
    decoded_sentences = [sp.decode_ids(tokens) for tokens in tokenized_data]
    return decoded_sentences

def save_decoded_sentences(decoded_sentences, output_file_path):
    with open(output_file_path, 'w') as f:
        for sentence in decoded_sentences:
            f.write(sentence + '\n')

    print(f"Decoded sentences have been written to '{output_file_path}'.")

if __name__ == "__main__":
    sp = spm.SentencePieceProcessor()
    sp.load('m.model')

    tokenized_data = load_token_ids('insert file name')

    decoded_sentences = decode_token_ids(tokenized_data, sp)

    save_decoded_sentences(decoded_sentences, 'output file')
