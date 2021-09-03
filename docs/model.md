
# Model 

The model is based on the research paper [Deep Neural Net with Attention for Multi-channel Multi-touch Attribution](https://arxiv.org/pdf/1809.02230.pdf)

## Architecture

### Input layer

name: input_layer
shape: (None, journey_max_len, nb_campaigns)

### Embedding layer

name: embedding_layer
input: input_layer
type: dense
activation: linear
output_shape: (None, journey_max_len, nb_units_embedding)

### LSTM layer

name: lstm_layer
input: embedding_layer
type: lstm
activation: tanh
output_shape: (None, journey_max_len, nb_units_lstm)

### Attention layer

name: attention_layer
input: lstm_layer
type: dense -> flatten -> activation
activation: softmax
output_shape: (None, journey_max_len)

### Weighted activation layer

name: weighted_activation_layer
input: lstm_layer, attention_layer
type: repeat_vector, permute, multiply, lambda
activation: none
output_shape: (None, nb_lstm_units)

### Output layer

name: output_layer
input: weighted_activation_layer
type: dense
activation: softmax
output_shape: (None, 1)

## Optimizer

Adam optimizer with learning_rate as hyperparameters, beta_1 set to 0.9, beta_2 set to 0.999 and epsilon set to 1e-7

## Attention model

The attention or attribution model is obtained by cutting the original model once trained such as the input is the input_layer and the output is the attention_layer.






