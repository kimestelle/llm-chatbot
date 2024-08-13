import os
import sentencepiece as spm

def process_file():
    if os.path.isfile('plato.txt'):
        spm.SentencePieceTrainer.Train('--input=plato.txt --model_prefix=m --vocab_size=20000 --split_by_whitespace=False')

        sp = spm.SentencePieceProcessor()
        sp.load('m.model')

        with open('processed_plato_as_ids.txt', 'w') as output_file:
            with open('plato.txt', 'r') as file:
                for line in file:
                    processed_line = sp.encode_as_ids(line.strip())
                    output_file.write(' '.join(map(str, processed_line)) + '\n')

        print("Processed sentences as IDs have been written to 'processed_plato_as_ids.txt'.")

    else:
        print("File 'plato.txt' not found.")

def get_pad_id():
    sp = spm.SentencePieceProcessor()
    sp.load('m.model')
    return sp.pad_id()

if __name__ == "__main__":
    process_file()
