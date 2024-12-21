#importing libraries
import speech_recognition as sr
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
from speechbrain.inference.speaker import SpeakerRecognition
import editdistance
from vosk import Model, KaldiRecognizer
import wave
import json
import torch

class verify:
    def __init__(self, original: str, test: str, model_path: str):
        """Takes the full path of the original record 
        and the one to be tested along with the path
        oof the model

        Args:
            original (str): Path of the original record
            test (str): Path of the record to be test
        """
        self.original = original
        self.test = test
        self.model = Model(model_path)
    
    def preprocess_audio(self, path: str, target_sample_rate: int = 16000) -> str:
        """
        Preprocesses the audio file to ensure it is mono PCM WAV with the target sample rate.

        Args:
            path (str): The full path of the original audio file.
            target_sample_rate (int): The desired sampling rate (default is 16kHz).

        Returns:
            str: Path to the preprocessed audio file.
        """
        audio = AudioSegment.from_file(path)

        # Convert to mono
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Resample to the target sample rate
        if audio.frame_rate != target_sample_rate:
            audio = audio.set_frame_rate(target_sample_rate)

        # Save to a temporary file
        temp_path = "temp_audio.wav"
        audio.export(temp_path, format="wav")
        return temp_path
    
    def transcribe_audio(self, path: str) -> str:
        """
        Transcribe speech from an audio file using Vosk.

        Args:
            path (str): The full path of the audio file.

        Returns:
            str: The extracted text from the speech.
        """
        # Preprocess the audio file
        processed_path = self.preprocess_audio(path)

        # Open the preprocessed audio file
        with wave.open(processed_path, "rb") as wf:
            # Initialize recognizer
            recognizer = KaldiRecognizer(self.model, wf.getframerate())
            recognizer.SetWords(True)

            # Transcribe audio
            text = ""
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text += result.get("text", "") + " "

            # Finalize transcription
            final_result = json.loads(recognizer.FinalResult())
            text += final_result.get("text", "")

        # Clean up temporary file
        os.remove(processed_path)

        return text.strip()
    
    def verification(self, threshold: float) -> bool:
        """This is where the test voice will be compared 
        with the original one to ensure it's from the same
        source using the cosine similarity calculated from
        form the embeddings

        Args:
            threshold (float): The value for which the two sounds will be considered similar
            in the souce or not (The higher the value is, the more accurate the verification is)
            and it's a flaot value between 0 and 1
        
        Returns:
           bool: True in case of a match, otherwise False
        """
        verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
        score, prediction = verification.verify_files(self.original, self.test) 

        if score >= threshold:
            return True
        else:
            return False
        
    def run(self, threshold: float) -> bool:
        """The main run method of the class
        to verify the secret phrase and 
        voice print of the speaker
        
        Args:
            threshold (float): The value for which the two texts extracted from the original and test
            will be considered similar in the souce or not (The higher the value is, the more accurate 
            the verification is) and it's a flaot value between 0 and 1.

        Returns:
            bool: True in case of a match, otherwise False
        """
        
        #Text of original sound 
        text_of_original = self.transcribe_audio(self.original)
        
        #Text of test sound
        text_of_test = self.transcribe_audio(self.test)
        
        #This calculates the edit distance between the two texts of the original sound
        #and the test sound, which is the number of moves to make the two texts similar
        number_of_moves_to_be_similar = editdistance.eval(text_of_test, text_of_original)
        
        #Percentage of similarty
        percentage = 100.0 - (float(number_of_moves_to_be_similar) / float(len(text_of_original)) * 100.0)
        
        if (percentage >= threshold) and self.verification(threshold):
            return True
        else:
            return False
        