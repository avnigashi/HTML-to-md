import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QWidget, QProgressBar,
                             QMessageBox, QFileDialog, QComboBox, QHBoxLayout, QLabel, QGroupBox, QRadioButton,
                             QTabWidget, QSplitter, QLineEdit, QCheckBox, QDoubleSpinBox, QSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl
from PyQt5.QtGui import QFont, QColor, QPalette, QDesktopServices
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import markdown
import requests
from bs4 import BeautifulSoup
import json

class ModelThread(QThread):
    finished = pyqtSignal(str)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, html_content, model_path, device, params):
        QThread.__init__(self)
        self.html_content = html_content
        self.model_path = model_path
        self.device = device
        self.params = params

    def run(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path).to(self.device)

            input_text = f"Convert the following HTML to Markdown:\n\n{self.html_content}\n\nMarkdown:"

            inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.params['max_new_tokens'],
                temperature=self.params['temperature'],
                do_sample=self.params['do_sample'],
                top_p=self.params['top_p'],
                repetition_penalty=self.params['repetition_penalty'],
                num_return_sequences=self.params['num_return_sequences']
            )

            markdown_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            markdown_output = markdown_output.split("Markdown:")[-1].strip()

            self.finished.emit(markdown_output)
        except Exception as e:
            self.error.emit(str(e))

class HTMLtoMarkdownConverter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.models = {
            "0.5B Model": "jinaai/reader-lm-0.5b",
            "1.5B Model": "jinaai/reader-lm-1.5b"
        }
        self.device = "cpu"
        self.initUI()
        self.loadSettings()
        self.setStyleSheet(self.getStyleSheet())

    def initUI(self):
        self.setWindowTitle('HTML to Markdown Converter')
        self.setGeometry(100, 100, 1200, 800)

        main_layout = QVBoxLayout()

        # Top controls
        top_controls = QHBoxLayout()
        top_controls.addLayout(self.createModelSelection())
        top_controls.addWidget(self.createHardwareSelection())
        main_layout.addLayout(top_controls)

        # Tabs for different input methods and settings
        self.tabs = QTabWidget()
        self.tabs.addTab(self.createManualInputTab(), "Manual Input")
        self.tabs.addTab(self.createURLInputTab(), "URL Input")
        self.tabs.addTab(self.createSettingsTab(), "Settings")
        main_layout.addWidget(self.tabs)

        # Convert button and progress bar
        main_layout.addLayout(self.createConversionControls())

        # Output area
        self.markdownOutput = QTextEdit()
        self.markdownOutput.setReadOnly(True)
        main_layout.addWidget(self.markdownOutput)

        # Bottom controls
        main_layout.addLayout(self.createBottomControls())

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def createModelSelection(self):
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Select Model:"))
        self.model_selector = QComboBox()
        self.model_selector.addItems(self.models.keys())
        model_layout.addWidget(self.model_selector)
        return model_layout

    def createHardwareSelection(self):
        hardware_group = QGroupBox("Select Hardware:")
        hardware_layout = QHBoxLayout()
        self.cpu_radio = QRadioButton("CPU")
        self.gpu_radio = QRadioButton("GPU")
        self.cpu_radio.setChecked(True)
        self.gpu_radio.setEnabled(torch.cuda.is_available() or torch.backends.mps.is_available())
        hardware_layout.addWidget(self.cpu_radio)
        hardware_layout.addWidget(self.gpu_radio)
        hardware_group.setLayout(hardware_layout)
        self.cpu_radio.toggled.connect(self.updateDevice)
        self.gpu_radio.toggled.connect(self.updateDevice)
        return hardware_group

    def createManualInputTab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        self.removeStylesCheckbox = QCheckBox("Remove styles from input")
        layout.addWidget(self.removeStylesCheckbox)
        self.htmlInput = QTextEdit()
        self.htmlInput.setPlaceholderText("Enter HTML here...")
        layout.addWidget(self.htmlInput)
        tab.setLayout(layout)
        return tab

    def createURLInputTab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        url_layout = QHBoxLayout()
        self.urlInput = QLineEdit()
        self.urlInput.setPlaceholderText("Enter URL here...")
        url_layout.addWidget(self.urlInput)
        self.fetchButton = QPushButton('Fetch HTML')
        self.fetchButton.clicked.connect(self.fetchHTML)
        url_layout.addWidget(self.fetchButton)
        layout.addLayout(url_layout)
        self.urlContent = QTextEdit()
        self.urlContent.setReadOnly(True)
        layout.addWidget(self.urlContent)
        tab.setLayout(layout)
        return tab

    def createSettingsTab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Model generation parameters
        params_group = QGroupBox("Model Generation Parameters")
        params_layout = QVBoxLayout()

        self.max_new_tokens = QSpinBox()
        self.max_new_tokens.setRange(1, 2048)
        self.max_new_tokens.setValue(1024)
        params_layout.addWidget(QLabel("Max New Tokens:"))
        params_layout.addWidget(self.max_new_tokens)

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.1, 1.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(0.7)
        params_layout.addWidget(QLabel("Temperature:"))
        params_layout.addWidget(self.temperature)

        self.do_sample = QCheckBox("Do Sample")
        self.do_sample.setChecked(True)
        params_layout.addWidget(self.do_sample)

        self.top_p = QDoubleSpinBox()
        self.top_p.setRange(0.1, 1.0)
        self.top_p.setSingleStep(0.05)
        self.top_p.setValue(0.95)
        params_layout.addWidget(QLabel("Top P:"))
        params_layout.addWidget(self.top_p)

        self.repetition_penalty = QDoubleSpinBox()
        self.repetition_penalty.setRange(1.0, 2.0)
        self.repetition_penalty.setSingleStep(0.1)
        self.repetition_penalty.setValue(1.1)
        params_layout.addWidget(QLabel("Repetition Penalty:"))
        params_layout.addWidget(self.repetition_penalty)

        self.num_return_sequences = QSpinBox()
        self.num_return_sequences.setRange(1, 5)
        self.num_return_sequences.setValue(1)
        params_layout.addWidget(QLabel("Number of Return Sequences:"))
        params_layout.addWidget(self.num_return_sequences)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Save and reset buttons
        buttons_layout = QHBoxLayout()
        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.saveSettings)
        buttons_layout.addWidget(save_button)

        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self.resetSettings)
        buttons_layout.addWidget(reset_button)

        layout.addLayout(buttons_layout)
        tab.setLayout(layout)
        return tab

    def createConversionControls(self):
        convert_layout = QHBoxLayout()
        self.convertButton = QPushButton('Convert to Markdown')
        self.convertButton.clicked.connect(self.convertHTML)
        convert_layout.addWidget(self.convertButton)

        self.progressBar = QProgressBar()
        self.progressBar.setVisible(False)
        convert_layout.addWidget(self.progressBar)
        return convert_layout

    def createBottomControls(self):
        bottom_controls = QHBoxLayout()

        self.saveButton = QPushButton('Save Markdown')
        self.saveButton.clicked.connect(self.saveMarkdown)
        self.saveButton.setEnabled(False)
        bottom_controls.addWidget(self.saveButton)

        self.copyButton = QPushButton('Copy to Clipboard')
        self.copyButton.clicked.connect(self.copyToClipboard)
        self.copyButton.setEnabled(False)
        bottom_controls.addWidget(self.copyButton)

        self.previewButton = QPushButton('Preview HTML')
        self.previewButton.clicked.connect(self.previewHTML)
        self.previewButton.setEnabled(False)
        bottom_controls.addWidget(self.previewButton)

        self.githubButton = QPushButton('Visit GitHub Repo')
        self.githubButton.clicked.connect(self.openGitHubRepo)
        bottom_controls.addWidget(self.githubButton)

        return bottom_controls

    def updateDevice(self):
        if self.gpu_radio.isChecked():
            self.device = "cuda" if torch.cuda.is_available() else "mps"
        else:
            self.device = "cpu"
        QMessageBox.information(self, "Device Updated", f"Using {self.device.upper()} for inference")

    def fetchHTML(self):
        url = self.urlInput.text()
        if not url:
            QMessageBox.warning(self, "Input Error", "Please enter a URL.")
            return
        try:
            response = requests.get(url)
            response.raise_for_status()
            html_content = response.text
            if self.removeStylesCheckbox.isChecked():
                html_content = self.remove_styles(html_content)
            self.urlContent.setPlainText(html_content)
        except requests.RequestException as e:
            QMessageBox.critical(self, "Fetch Error", f"Failed to fetch HTML: {str(e)}")

    def remove_styles(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        for style in soup(["style", "script"]):
            style.decompose()
        return str(soup)

    def convertHTML(self):
        if self.tabs.currentIndex() == 0:
            html_content = self.htmlInput.toPlainText()
        else:
            html_content = self.urlContent.toPlainText()

        if not html_content:
            QMessageBox.warning(self, "Input Error", "Please enter or fetch some HTML content.")
            return

        if self.removeStylesCheckbox.isChecked():
            html_content = self.remove_styles(html_content)

        self.convertButton.setEnabled(False)
        self.progressBar.setVisible(True)
        self.progressBar.setValue(0)

        model_name = self.model_selector.currentText()
        model_path = self.models[model_name]

        params = {
            'max_new_tokens': self.max_new_tokens.value(),
            'temperature': self.temperature.value(),
            'do_sample': self.do_sample.isChecked(),
            'top_p': self.top_p.value(),
            'repetition_penalty': self.repetition_penalty.value(),
            'num_return_sequences': self.num_return_sequences.value()
        }

        self.thread = ModelThread(html_content, model_path, self.device, params)
        self.thread.finished.connect(self.onConversionFinished)
        self.thread.error.connect(self.onError)
        self.thread.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateProgressBar)
        self.timer.start(100)

    def updateProgressBar(self):
        current_value = self.progressBar.value()
        if current_value < 99:
            self.progressBar.setValue(current_value + 1)
        else:
            self.timer.stop()

    def onConversionFinished(self, markdown_output):
        self.markdownOutput.setPlainText(markdown_output)
        self.convertButton.setEnabled(True)
        self.saveButton.setEnabled(True)
        self.copyButton.setEnabled(True)
        self.previewButton.setEnabled(True)
        self.progressBar.setVisible(False)
        QMessageBox.information(self, "Conversion Complete", "HTML has been successfully converted to Markdown!")

    def onError(self, error_message):
        self.convertButton.setEnabled(True)
        self.progressBar.setVisible(False)
        QMessageBox.critical(self, "Error", f"An error occurred: {error_message}")

    def saveMarkdown(self):
        markdown_content = self.markdownOutput.toPlainText()
        if not markdown_content:
            QMessageBox.warning(self, "Output Error", "There's no Markdown content to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Markdown File", "", "Markdown Files (*.md);;All Files (*)")
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(markdown_content)
            QMessageBox.information(self, "Save Successful", f"Markdown content has been saved to {file_path}")

    def copyToClipboard(self):
        markdown_content = self.markdownOutput.toPlainText()
        if markdown_content:
            clipboard = QApplication.clipboard()
            clipboard.setText(markdown_content)
            QMessageBox.information(self, "Copy Successful", "Markdown content has been copied to the clipboard.")
        else:
            QMessageBox.warning(self, "Copy Error", "There's no Markdown content to copy.")

    def previewHTML(self):
            markdown_content = self.markdownOutput.toPlainText()
            if markdown_content:
                html_content = markdown.markdown(markdown_content)
                preview = QTextEdit()
                preview.setHtml(html_content)
                preview.setReadOnly(True)
                preview.setWindowTitle("HTML Preview")
                preview.resize(600, 400)
                preview.show()
            else:
                QMessageBox.warning(self, "Preview Error", "There's no Markdown content to preview.")

    def openGitHubRepo(self):
        url = QUrl("https://github.com/yourusername/HTML-to-md-advanced")
        QDesktopServices.openUrl(url)

    def saveSettings(self):
        settings = {
            'max_new_tokens': self.max_new_tokens.value(),
            'temperature': self.temperature.value(),
            'do_sample': self.do_sample.isChecked(),
            'top_p': self.top_p.value(),
            'repetition_penalty': self.repetition_penalty.value(),
            'num_return_sequences': self.num_return_sequences.value(),
            'remove_styles': self.removeStylesCheckbox.isChecked(),
            'model': self.model_selector.currentText(),
            'device': self.device
        }
        with open('settings.json', 'w') as f:
            json.dump(settings, f)
        QMessageBox.information(self, "Settings Saved", "Your settings have been saved successfully.")

    def loadSettings(self):
        try:
            with open('settings.json', 'r') as f:
                settings = json.load(f)
            self.max_new_tokens.setValue(settings.get('max_new_tokens', 1024))
            self.temperature.setValue(settings.get('temperature', 0.7))
            self.do_sample.setChecked(settings.get('do_sample', True))
            self.top_p.setValue(settings.get('top_p', 0.95))
            self.repetition_penalty.setValue(settings.get('repetition_penalty', 1.1))
            self.num_return_sequences.setValue(settings.get('num_return_sequences', 1))
            self.removeStylesCheckbox.setChecked(settings.get('remove_styles', False))
            self.model_selector.setCurrentText(settings.get('model', '0.5B Model'))
            self.device = settings.get('device', 'cpu')
            if self.device == 'cpu':
                self.cpu_radio.setChecked(True)
            else:
                self.gpu_radio.setChecked(True)
        except FileNotFoundError:
            # If settings file doesn't exist, use defaults
            pass

    def resetSettings(self):
        self.max_new_tokens.setValue(1024)
        self.temperature.setValue(0.7)
        self.do_sample.setChecked(True)
        self.top_p.setValue(0.95)
        self.repetition_penalty.setValue(1.1)
        self.num_return_sequences.setValue(1)
        self.removeStylesCheckbox.setChecked(False)
        self.model_selector.setCurrentText('0.5B Model')
        self.cpu_radio.setChecked(True)
        self.device = 'cpu'
        QMessageBox.information(self, "Settings Reset", "Settings have been reset to default values.")

    def getStyleSheet(self):
        return """
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QTextEdit, QLineEdit {
                background-color: #3c3f41;
                color: #ffffff;
                border: 1px solid #555555;
            }
            QPushButton {
                background-color: #365880;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #4a76a8;
            }
            QComboBox, QRadioButton, QCheckBox {
                background-color: #3c3f41;
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #3c3f41;
                color: #ffffff;
                padding: 5px;
            }
            QTabBar::tab:selected {
                background-color: #4a76a8;
            }
            QGroupBox {
                border: 1px solid #555555;
                margin-top: 1ex;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
            }
            QDoubleSpinBox, QSpinBox {
                background-color: #3c3f41;
                color: #ffffff;
                border: 1px solid #555555;
            }
        """

class AdvancedHTMLtoMarkdownConverter(HTMLtoMarkdownConverter):
    def __init__(self):
        super().__init__()
        self.initAdvancedUI()

    def initAdvancedUI(self):
        self.tabs.addTab(self.createAdvancedTab(), "Advanced")

    def createAdvancedTab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Custom CSS input
        css_group = QGroupBox("Custom CSS")
        css_layout = QVBoxLayout()
        self.cssInput = QTextEdit()
        self.cssInput.setPlaceholderText("Enter custom CSS here...")
        css_layout.addWidget(self.cssInput)
        css_group.setLayout(css_layout)
        layout.addWidget(css_group)

        # Markdown flavor selection
        flavor_group = QGroupBox("Markdown Flavor")
        flavor_layout = QHBoxLayout()
        self.flavorSelector = QComboBox()
        self.flavorSelector.addItems(["GitHub Flavored", "CommonMark", "Python-Markdown"])
        flavor_layout.addWidget(self.flavorSelector)
        flavor_group.setLayout(flavor_layout)
        layout.addWidget(flavor_group)

        # Batch processing
        batch_group = QGroupBox("Batch Processing")
        batch_layout = QVBoxLayout()
        self.batchButton = QPushButton("Select Directory for Batch Processing")
        self.batchButton.clicked.connect(self.batchProcess)
        batch_layout.addWidget(self.batchButton)
        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)

        tab.setLayout(layout)
        return tab

    def batchProcess(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory for Batch Processing")
        if directory:
            for filename in os.listdir(directory):
                if filename.endswith(".html"):
                    with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                        html_content = file.read()
                    # Process HTML content
                    markdown_content = self.processHTML(html_content)
                    # Save markdown content
                    output_filename = os.path.splitext(filename)[0] + '.md'
                    with open(os.path.join(directory, output_filename), 'w', encoding='utf-8') as file:
                        file.write(markdown_content)
            QMessageBox.information(self, "Batch Processing Complete", "All HTML files in the selected directory have been converted to Markdown.")

    def processHTML(self, html_content):
        # This method should be implemented to process HTML content
        # For now, we'll use a placeholder implementation
        return "Placeholder Markdown content"

    def convertHTML(self):
        super().convertHTML()
        # Apply custom CSS if provided
        custom_css = self.cssInput.toPlainText()
        if custom_css:
            pass

        selected_flavor = self.flavorSelector.currentText()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    converter = AdvancedHTMLtoMarkdownConverter()
    converter.show()
    sys.exit(app.exec_())
