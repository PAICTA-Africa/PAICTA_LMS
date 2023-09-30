import json
from tkinter import scrolledtext
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import tkinter as tk

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create the Tkinter GUI window
root = tk.Tk()
root.title("Algebra Question Answering")

# Define the train_model() function
def train_model():
    # Load the training data from train.json
    with open('train.json') as train_file:
        train_data = json.load(train_file)

    # Preprocess the training data
    train_questions = [data['question'] for data in train_data for _ in data['steps']]
    train_steps = [step for data in train_data for step in data['steps']]

    # Initialize the BERT model
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

    # Tokenize and encode the training data
    train_inputs = tokenizer.batch_encode_plus(
        list(zip(train_questions, train_steps)),
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    train_input_ids = train_inputs['input_ids']
    train_attention_masks = train_inputs['attention_mask']

    # Convert the answer tensor to Long type
    train_start_positions = torch.tensor(
        [tokenizer.encode(step, add_special_tokens=False)[0] for step in train_steps],
        dtype=torch.long
    )
    train_end_positions = torch.tensor(
        [tokenizer.encode(step, add_special_tokens=False)[-1] for step in train_steps],
        dtype=torch.long
    )

    # Train the model
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(3):
        optimizer.zero_grad()
        outputs = model(
            input_ids=train_input_ids,
            attention_mask=train_attention_masks,
            start_positions=train_start_positions,
            end_positions=train_end_positions
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Save the trained model
    model.save_pretrained('trained_model')

    # Show completion message
    completion_label.config(text="Training Completed!")

# Define the predict_answers() function
def predict_answers():
    # Load the trained model
    model = BertForQuestionAnswering.from_pretrained('trained_model')

    # Load the prediction data from predict.json
    with open('predict.json') as predict_file:
        predict_data = json.load(predict_file)

    # Preprocess the prediction data
    predict_questions = [data['question'] for data in predict_data]

    # Tokenize and encode the prediction data
    predict_inputs = tokenizer.batch_encode_plus(
        predict_questions,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    predict_input_ids = predict_inputs['input_ids']
    predict_attention_masks = predict_inputs['attention_mask']

    # Predict answers for the questions
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=predict_input_ids,
            attention_mask=predict_attention_masks
        )
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

    # Extract the predicted answers
    predicted_answers = []
    for i in range(len(predict_data)):
        start_index = torch.argmax(start_logits[i])
        end_index = torch.argmax(end_logits[i])
        predicted_answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(predict_input_ids[i][start_index:end_index+1])
        )
        predicted_answers.append(predicted_answer)

    # Clear existing labels, if any
    for widget in root.winfo_children():
        widget.destroy()

    # Show the predicted answers
    for i, data in enumerate(predict_data):
        question_label = tk.Label(root, text='Question: ' + data['question'])
        question_label.pack()
        for j, step in enumerate(data['steps']):
            step_label = tk.Label(root, text='Step ' + str(j + 1) + ': ' + step)
            step_label.pack()
        answer_label = tk.Label(root, text='Predicted Answer: ' + predicted_answers[i])
        answer_label.pack()







# Load the prediction data from predict.json
with open('predict.json', 'r') as predict_file:
    predict_data = json.load(predict_file)

# Create a function to retrieve a step-by-step answer
def get_step_by_step_answer():
    # Get the user's question from the input field
    user_question = question_entry.get().strip()

    # Search for the user's question in the prediction data
    answer_found = False
    for data in predict_data:
        if data['question'].strip().lower() == user_question.lower():
            step_by_step_answer = "\n".join(data['steps'])
            answer_text.config(text="Step-by-Step Answer:\n" + step_by_step_answer)
            answer_found = True
            break

    if not answer_found:
        answer_text.config(text="No step-by-step answer found for this question.")

# Create the main application window
root = tk.Tk()
root.title("Algebra Step-by-Step Solver")

# Create a label and an entry widget for the user to enter a question
question_label = tk.Label(root, text="Enter your algebra question:")
question_label.pack(padx=20, pady=10)
question_entry = tk.Entry(root, width=50)
question_entry.pack(padx=20, pady=10)

# Create a button to submit the question
submit_button = tk.Button(root, text="Submit", command=get_step_by_step_answer)
submit_button.pack(pady=10)

# Create a label to display the step-by-step answer
answer_text = tk.Label(root, text="", wraplength=400)
answer_text.pack(padx=20, pady=10)

# Start the Tkinter event loop
root.mainloop()