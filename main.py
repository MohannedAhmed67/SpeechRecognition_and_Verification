from sound_recognition_and_verification import verify

if __name__ == '__main__':
    #Path of the original record
    original = "original.wav"
    
    #Path of the test record
    test = "speech.wav"
    
    #Path of the model
    model = "vosk-model-small-en-us-0.15"
    
    doorlock_verification = verify(original, test, model)
    
    if doorlock_verification.run(0.8):
        print("This is a match!")
    else:
        print("Please try again :(")