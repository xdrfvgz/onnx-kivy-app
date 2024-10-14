
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
import onnxruntime as ort
import numpy as np
from transformers import BertTokenizer

class MyApp(App):
    def build(self):
        self.session = None

        self.box = BoxLayout(orientation='vertical')
        self.input = TextInput(hint_text='Enter text here')
        self.button = Button(text='Run Inference', on_press=self.on_button_press)
        self.output = Label(text='Inference result will appear here')
        self.load_button = Button(text='Load Model', on_press=self.show_file_chooser)

        self.box.add_widget(self.input)
        self.box.add_widget(self.button)
        self.box.add_widget(self.load_button)
        self.box.add_widget(self.output)

        return self.box

    def show_file_chooser(self, instance):
        content = BoxLayout(orientation='vertical')
        self.file_chooser = FileChooserListView(filters=['*.onnx'])
        content.add_widget(self.file_chooser)

        select_button = Button(text='Select', size_hint_y=None, height=50)
        select_button.bind(on_press=self.select_model_file)
        content.add_widget(select_button)

        self.file_popup = Popup(title="Choose Model File", content=content, size_hint=(0.9, 0.9))
        self.file_popup.open()

    def select_model_file(self, instance):
        if self.file_chooser.selection:
            model_path = self.file_chooser.selection[0]
            self.session = self.load_model(model_path)
            self.output.text = f"Model loaded: {model_path}"
        self.file_popup.dismiss()

    def load_model(self, model_path):
        session = ort.InferenceSession(model_path)
        return session

    def run_inference(self, input_text):
        if not self.session:
            self.output.text = "No model loaded."
            return None

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        inputs = tokenizer(input_text, return_tensors="np")
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        input_name1 = self.session.get_inputs()[0].name
        input_name2 = self.session.get_inputs()[1].name
        output_name = self.session.get_outputs()[0].name
        result = self.session.run([output_name], {input_name1: input_ids, input_name2: attention_mask})
        return result

    def on_button_press(self, instance):
        input_text = self.input.text
        result = self.run_inference(input_text)
        if result:
            self.output.text = f"Inference result: {result}"

if __name__ == '__main__':
    MyApp().run()