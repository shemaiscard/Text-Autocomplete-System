# Text-Autocomplete-System
Autocomplete System in PyQt6

Short Description:
This project is a Data Structures and Algorithms project that implements an autocomplete system using the Trie data structure in PyQt6. The system provides real-time word suggestions based on a list of words loaded from a file.

Features:
- Autocomplete suggestions for typed words based on a list from 'words.txt'.
- Displays suggestions in a list and allows users to select a suggestion to complete their input.
- Real-time suggestions as the user types.

Requirements:
- Python 3.x
- PyQt6
- A 'words.txt' file containing words (one word per line) for autocompletion.

Setup and Installation:
1. Install the required Python libraries:
   pip install PyQt6

2. Prepare a 'words.txt' file with a list of words. Each word should be on a new line. Example:
   apple
   banana
   orange
   pear
   grape

3. Save the Python script in the same directory as 'words.txt'.

4. Run the Python script:
   python autocomplete_system.py

How it Works:
- The program uses a Trie data structure to store the words from 'words.txt'.
- As you type in the input field, the system will automatically search for words that start with the same letters as the last word typed.
- It will display matching suggestions in a list below the input field.
- You can click a suggestion to complete your word.

File Structure:
.
├── autocomplete_system.py  # Main Python script
└── words.txt              # List of words for autocompletion

License:
This project is open source and free to use. You can modify it as per your requirements.
