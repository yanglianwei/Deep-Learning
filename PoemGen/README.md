# Long Short Term Memory (LSTM) for Text Generation

Utilize LSTM to generate Chinese Tang poem.

## Database: A Brief Introduction

The database has been preprocessed and contains 57580 Chinese Tang poem. Each poem is limited to 125 words, starting with `<START>` and ending with `<EOP>` . `</s>` is used as placeholders if the poem is shorter than 125 words.

The database is saved in `npz` format and composed of parts:
* `data`: poems, expressed in integers in the word dictionary
* `ix2word`: mapping integers to Chinese characters
* `word2ix`: mapping Chinese characters to integers

## RNN structure: Long Short Term Memory

The structure is a general LSTM model, check this blog for more details:

[Understanding-LSTMs from colah's blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

### 1. Construct word embeddings

    self.embeddings = nn.Embedding(vocab_size, embedding_dim)

### 2. LSTM

    self.lstm = nn.LSTM(embedding_dim, hidden_size=self.hidden_dim,
                            num_layers=NUM_LAYERS, batch_first=False)
* Inputs:

  - input (_seq_len_, _batch_, _input_size_)

  - h_0 (_num_layers_ * _num_directions_, _batch_, _hidden_size_)

  - c_0 (_num_layers_ * _num_directions_, _batch_, _hidden_size_)

* Oututs:

  - output (_seq_len_, _batch_, _num_directions_ * _hidden_size_)

  - h_n (same as h_0)

  - c_n (same as c_0)


### 3. FC layer

    self.linear = nn.Linear(self.hidden_dim, vocab_size)


## Applications
* Set several first characters
      
      start_chars = '雁栖'
      poem_length = 48
      results = generate(model, start_chars, poem_length, ix2word, word2ix, device)
      print(''.join(i for i in results))

      雁栖有何人，时有平阳客。
      自古古来书，有诗多不饱。
      我生西之间，君子不相识。
      一见如来人，一日无所欲。

* Set start characters for each sentense

      initial_chars = '端午安康'
      poem_style = 5
      results = generate_initials(model, initial_chars, poem_style, ix2word, word2ix, device)
      print(''.join(i for i in results))

      端拙须长叹，
      午年何所为。
      安贫即此路，
      康乐当及时。