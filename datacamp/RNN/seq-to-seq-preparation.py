## part 1

# Get maximum length of the sentences
pt_length = max([len(sentence.split(' ')) for sentence in pt_sentences])

# Transform text to sequence of numerical indexes
X = input_tokenizer.texts_to_sequences(pt_sentences)

# Pad the sequences
X = pad_sequences(X, maxlen=pt_length, padding='post')

# Print first sentence
print(pt_sentences[0])

# Print transformed sentence
print(X[0])

## part 2

# Initialize the variable
Y = transform_text_to_sequences(en_sentences, output_tokenizer)

# Temporary list
ylist = list()
for sequence in Y:
  	# One-hot encode sentence and append to list
    ylist.append(to_categorical(sequence, num_classes=en_vocab_size))

# Update the variable
Y = np.array(ylist).reshape(Y.shape[0], Y.shape[1], en_vocab_size)

# Print the raw sentence and its transformed version
print("Raw sentence: {0}\nTransformed: {1}".format(en_sentences[0], Y[0]))

## part 3

# Function to predict many phrases
def predict_many(model, sentences, index_to_word, raw_dataset):
    for i, sentence in enumerate(sentences):
        # Translate the Portuguese sentence
        translation = predict_one(model, sentence, index_to_word)
        
        # Get the raw Portuguese and English sentences
        raw_target, raw_src = raw_dataset[i]
        
        # Print the correct Portuguese and English sentences and the predicted
        print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))

predict_many(model, X_test[:10], en_index_to_word, test)