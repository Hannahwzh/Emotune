from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import re
import random
from collections import defaultdict
import spacy
import numpy as np
from datetime import datetime

# Load models (this might take some time)
print("Loading models... This may take a moment.")
chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Use smaller bart model for baseline summary generation before customization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load SpaCy for NLP tasks
try:
    nlp = spacy.load("en_core_web_sm")
    print("SpaCy model loaded successfully!")
except:
    print("Installing SpaCy model...")
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

print("All models loaded successfully!")

class EmotionalStoryChatbot:
    def __init__(self):
        self.chat_history = []
        self.emotion_history = []
        self.story_segments = []
        self.conversation_turns = 0
        self.mentioned_topics = set()
        self.entities = {
            "PERSON": [],
            "DATE": [],
            "TIME": [],
            "LOC": [],
            "GPE": [],  # Geopolitical entities (countries, cities)
            "ORG": [],  # Organizations
            "EVENT": []
        }
        
        # Song structures for lyrical summary
        self.song_templates = [
            {
                "title_template": "{dominant_emotion} {time_reference}",
                "verse_structure": ["who_verse", "what_verse", "where_verse", "emotion_verse"],
                "chorus_template": "In a world of {emotion_opposite}, their {dominant_emotion} shines through\n" +
                                  "The story of {main_person} will always be true\n" +
                                  "From {main_location} to the depths of their heart\n" +
                                  "This {dominant_emotion} tale sets them apart"
            },
            {
                "title_template": "The {dominant_emotion} of {main_person}",
                "verse_structure": ["where_verse", "who_verse", "emotion_verse", "what_verse"],
                "chorus_template": "{main_person} felt {dominant_emotion} {time_reference}\n" +
                                  "Their story echoes through time and space\n" +
                                  "With each word shared, a piece of their soul\n" +
                                  "Their {dominant_emotion} journey makes them whole"
            }
        ]
        
        # Emotion opposites for lyrical contrast
        self.emotion_opposites = {
            "joy": "sorrow",
            "sadness": "happiness",
            "anger": "calm",
            "fear": "courage",
            "surprise": "expectation",
            "love": "indifference",
            "neutral": "passion"
        }
        
        # Enhanced follow-up questions with emotion-specific prompts
        self.follow_up_by_emotion = {
            "joy": [
                "What was the most joyful part of that experience for you?",
                "Your happiness really comes through! What made that moment so special?",
                "That sounds wonderful! Did you share that happy moment with someone important to you?",
                "I can feel your excitement! What were you thinking when that happened?",
                "Those positive moments are so important. How did that experience change your perspective?"
            ],
            "sadness": [
                "I'm truly sorry you went through that. How have you been coping with these feelings?",
                "That sounds really difficult. What has helped you get through this challenging time?",
                "I can hear the sadness in your words. Would it help to talk more about how that affected you?",
                "Thank you for trusting me with something so personal. Was there anyone who supported you during this time?",
                "It takes courage to share painful experiences. How has this shaped who you are today?"
            ],
            "anger": [
                "That situation would frustrate anyone. What bothered you the most about what happened?",
                "I understand why you'd feel angry about that. Have you had a chance to express these feelings?",
                "That does sound unfair. What would justice or resolution look like for you in this situation?",
                "Your anger is completely valid. How did you respond when this happened?",
                "I appreciate you sharing these strong feelings. Has your perspective on this changed over time?"
            ],
            "fear": [
                "That sounds really frightening. What has helped you manage these worries?",
                "I understand why that would cause anxiety. What's your biggest concern about this situation?",
                "Those fears make a lot of sense. Have you found any ways to find comfort when these thoughts come up?",
                "Thank you for sharing something so vulnerable. How has this fear impacted your daily life?",
                "It's brave to face your fears by talking about them. What would help you feel safer?"
            ],
            "surprise": [
                "What an unexpected turn of events! How did you react in the moment?",
                "That must have caught you completely off guard! How did this surprise change things for you?",
                "I can imagine your shock! Did this unexpected situation lead to any positive discoveries?",
                "Life certainly has its surprises! Has this changed how you approach similar situations?",
                "That's quite the revelation! Have your feelings about this surprise changed since it happened?"
            ],
            "love": [
                "That relationship sounds really meaningful. What qualities do you value most about this connection?",
                "It's beautiful to hear about such deep feelings. How has this love transformed you?",
                "Those bonds are so precious. What moments have made you feel especially close to this person?",
                "Love stories are always unique. What makes this relationship particularly special to you?",
                "That connection sounds profound. Has this relationship helped you learn something about yourself?"
            ],
            "neutral": [
                "Could you tell me more about how you felt during that experience?",
                "That's interesting. What details of that situation stand out most in your memory?",
                "I'd love to hear more about your perspective on this. What were you thinking at the time?",
                "Thank you for sharing that. How significant was this event in your life story?",
                "I'm curious about your experience. How did this situation affect your relationships or outlook?"
            ]
        }
        
        # Advanced emotion-specific response templates
        self.emotion_responses = {
            "joy": [
                "I can feel your happiness radiating through your words! That's wonderful to hear. {follow_up}",
                "What a delightful experience! It's moments like these that make life beautiful. {follow_up}",
                "I'm genuinely happy for you! Those joyful moments are so precious. {follow_up}",
                "Your enthusiasm is contagious! Thank you for sharing such positive energy. {follow_up}",
                "That's truly something to celebrate! It sounds like a meaningful high point for you. {follow_up}"
            ],
            "sadness": [
                "I'm truly sorry you're going through this difficult time. Your feelings are completely valid. {follow_up}",
                "That sounds incredibly painful. I'm here to listen without judgment whenever you need to talk. {follow_up}",
                "My heart goes out to you. These emotional burdens can feel so heavy sometimes. {follow_up}",
                "Thank you for trusting me with something so personal and difficult. It takes courage to be vulnerable. {follow_up}",
                "I can hear how much this has affected you. Please know that you don't have to process these feelings alone. {follow_up}"
            ],
            "anger": [
                "I completely understand why you'd feel frustrated and angry about that situation. It sounds truly unfair. {follow_up}",
                "Your anger is absolutely justified based on what you've shared. I would feel the same way. {follow_up}",
                "That would test anyone's patience. It sounds like important boundaries were crossed. {follow_up}",
                "I appreciate you sharing these strong feelings. Sometimes anger is an appropriate response to injustice. {follow_up}",
                "That situation sounds incredibly frustrating. Your feelings are a natural response to being treated that way. {follow_up}"
            ],
            "fear": [
                "Those concerns sound really overwhelming. It's completely normal to feel anxious about something so important. {follow_up}",
                "I understand why that situation would trigger such worry. Uncertainty can be really difficult to navigate. {follow_up}",
                "Your fears make perfect sense given what you're facing. Please be gentle with yourself through this. {follow_up}",
                "That does sound quite daunting. Sometimes naming our fears can help us begin to process them. {follow_up}",
                "It takes courage to acknowledge these anxieties. Your concerns are completely valid. {follow_up}"
            ],
            "surprise": [
                "Wow! I can imagine how stunned you must have felt in that moment. Life can be so unpredictable. {follow_up}",
                "That's quite the unexpected development! Sometimes life's surprises completely reshape our perspective. {follow_up}",
                "I can practically see your wide eyes as you tell this! What an astonishing turn of events. {follow_up}",
                "That certainly wasn't what you were expecting! Those surprising moments often become powerful memories. {follow_up}",
                "How incredible! Those unexpected moments often make for the most compelling stories in our lives. {follow_up}"
            ],
            "love": [
                "The warmth in your words is palpable. Those deep connections truly give meaning to our lives. {follow_up}",
                "What a beautiful relationship you're describing. Those bonds of genuine affection are so precious. {follow_up}",
                "The fondness you feel really comes through in how you describe this. Those meaningful connections sustain us. {follow_up}",
                "That sounds like such a nurturing and significant relationship in your life. {follow_up}",
                "The love you're describing sounds like such a positive force in your life. {follow_up}"
            ],
            "neutral": [
                "Thank you for sharing that experience. I'm interested in understanding more about your perspective. {follow_up}",
                "That's quite insightful. I appreciate you taking the time to explain your thoughts on this. {follow_up}",
                "I see what you mean. Your perspective offers an interesting window into this situation. {follow_up}",
                "That's a thoughtful observation. I'm curious to learn more about your experiences. {follow_up}",
                "Thank you for sharing that story. It sounds like there's a lot to unpack there. {follow_up}"
            ]
        }
        
        # Opening messages to start conversation
        self.opening_messages = [
            "Hi there! I'd love to hear about how you're feeling today. Have any experiences been on your mind lately?",
            "Hello! I'm here to listen and chat about whatever's important to you right now. How has life been treating you lately?",
            "Hi! I'm really interested in hearing about your experiences and feelings. What's been happening in your world?",
            "Hello there! I'd love to get to know more about you and what matters in your life. How are you feeling today?",
            "Hi! I'm here to listen if there's anything you'd like to share about your life, thoughts, or feelings. What's on your mind?"
        ]
        
    def detect_entities(self, text):
        """Extract entities like people, places, dates from text using SpaCy"""
        doc = nlp(text)
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in self.entities:
                if ent.text not in self.entities[ent.label_]:
                    self.entities[ent.label_].append(ent.text)
        
        # Detect topics in text
        topics = []
        if any(word in text.lower() for word in ["mom", "dad", "sister", "brother", "parent", "family"]):
            topics.append("family")
        if any(word in text.lower() for word in ["job", "work", "career", "boss", "colleague"]):
            topics.append("work")
        if any(word in text.lower() for word in ["friend", "girlfriend", "boyfriend", "partner", "relationship", "marriage"]):
            topics.append("relationship")
        if any(word in text.lower() for word in ["sick", "health", "doctor", "therapy", "exercise", "diet"]):
            topics.append("health")
        if any(word in text.lower() for word in ["hobby", "game", "sport", "reading", "music", "art", "interest"]):
            topics.append("hobby")
        
        for topic in topics:
            self.mentioned_topics.add(topic)
        
        return topics
        
    def start_conversation(self):
        """Initialize conversation with an opening message"""
        opening = random.choice(self.opening_messages)
        self.chat_history.append({"role": "assistant", "content": opening})
        return opening
        
    def detect_emotion(self, text):
        if not text.strip():
            return "neutral", 1.0
        
        emotions = emotion_pipe(text)[0]
        top_emotion = max(emotions, key=lambda x: x['score'])
        return top_emotion['label'].lower(), top_emotion['score']
    
    def create_emotional_response(self, user_input, emotion):
        """Create a response that acknowledges the detected emotion and encourages sharing"""
        # Choose a response template based on the detected emotion
        if emotion not in self.emotion_responses:
            emotion = "neutral"
            
        # Extract entities and detect topics
        self.detect_entities(user_input)
            
        # Choose follow-up based on emotion
        follow_up = random.choice(self.follow_up_by_emotion[emotion])
        
        # Choose response template and fill in follow-up
        response_template = random.choice(self.emotion_responses[emotion])
        response = response_template.format(follow_up=follow_up)
        
        # For longer conversations, occasionally reference specific content
        if self.conversation_turns > 1 and len(user_input.split()) > 10 and random.random() > 0.5:
            # Try to extract a meaningful phrase
            sentences = re.split(r'[.!?]+', user_input)
            if sentences:
                important_phrase = random.choice([s for s in sentences if len(s.strip()) > 0])
                important_phrase = important_phrase.strip()
                if len(important_phrase.split()) > 3 and len(important_phrase) < 50:
                    response += f" I'm struck by what you said about '{important_phrase}'. That seems significant."
                
        return response
    
    def generate_response(self, user_input, emotion, emotion_score):
        # Track conversation
        self.chat_history.append({"role": "user", "content": user_input})
        self.emotion_history.append(emotion)
        
        # Add to story segments if meaningful content
        if len(user_input.split()) > 3:
            self.story_segments.append(user_input)
        
        # Create emotion-aware response
        response = self.create_emotional_response(user_input, emotion)
        
        # Track response
        self.chat_history.append({"role": "assistant", "content": response})
        self.conversation_turns += 1
        
        return response
    
    def create_lyrical_summary(self):
        """Create a song-like summary of the conversation based on who, what, when, where structure"""
        if len(self.story_segments) < 2:
            return "Not enough information shared to create a meaningful summary."
        
        # Combine story segments
        full_story = " ".join(self.story_segments)
        
        # Get base summary for content
        try:
            basic_summary = summarizer(full_story, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        except Exception as e:
            basic_summary = " ".join(self.story_segments[-3:])  # Fallback to recent messages
        
        # Get emotional profile
        emotion_counter = defaultdict(int)
        for emotion in self.emotion_history:
            emotion_counter[emotion] += 1
        
        dominant_emotion = max(emotion_counter.items(), key=lambda x: x[1])[0] if emotion_counter else "neutral"
        emotion_opposite = self.emotion_opposites.get(dominant_emotion, "indifference")
        
        # Select key entities for the song
        main_person = "they"
        if self.entities["PERSON"]:
            main_person = random.choice(self.entities["PERSON"])
        elif "I" in full_story or "me" in full_story.lower() or "my" in full_story.lower():
            main_person = "I"
            
        main_location = "somewhere"
        if self.entities["LOC"]:
            main_location = random.choice(self.entities["LOC"])
        elif self.entities["GPE"]:
            main_location = random.choice(self.entities["GPE"])
            
        time_reference = "today"
        if self.entities["DATE"]:
            time_reference = random.choice(self.entities["DATE"])
        elif self.entities["TIME"]:
            time_reference = random.choice(self.entities["TIME"])
        else:
            # Generate a poetic time reference if none found
            time_options = ["yesterday", "tomorrow", "in memories", "in moments", "through time"]
            time_reference = random.choice(time_options)
        
        # Choose a song template
        song_template = random.choice(self.song_templates)
        
        # Create song title
        title = song_template["title_template"].format(
            dominant_emotion=dominant_emotion.title(),
            main_person=main_person,
            time_reference=time_reference
        )
        
        # Create verses based on the structured data
        verses = {
            "who_verse": self._create_who_verse(main_person),
            "what_verse": self._create_what_verse(basic_summary),
            "where_verse": self._create_where_verse(main_location),
            "emotion_verse": self._create_emotion_verse(dominant_emotion, emotion_opposite)
        }
        
        # Assemble the song
        song = f"🎵 {title} 🎵\n\n"
        
        # Add verses in the order specified by the template
        for verse_type in song_template["verse_structure"]:
            song += verses[verse_type] + "\n\n"
            
        # Add chorus
        chorus = song_template["chorus_template"].format(
            dominant_emotion=dominant_emotion,
            emotion_opposite=emotion_opposite,
            main_person=main_person,
            main_location=main_location,
            time_reference=time_reference
        )
        
        song += "CHORUS:\n" + chorus + "\n\n"
        
        # Add a bridge with emotional journey
        emotional_journey = ", ".join([emotion.title() for emotion in self.emotion_history[-5:]])
        bridge = f"From {self.emotion_history[0] if self.emotion_history else 'neutral'} to {dominant_emotion}\n"
        bridge += f"Their journey flows through {emotional_journey}\n"
        bridge += f"With every word they shared their truth\n"
        bridge += f"A story of {dominant_emotion} that will never fade"
        
        song += "BRIDGE:\n" + bridge
        
        return song
        
    def _create_who_verse(self, main_person):
        """Create a verse about who was involved"""
        verse = f"Verse about WHO:\n"
        
        if main_person.lower() in ["i", "me", "they", "them"]:
            verse += f"A soul with stories yet untold\n"
            verse += f"Sharing pieces of their heart\n"
        else:
            verse += f"{main_person} with stories yet untold\n"
            verse += f"Sharing pieces of their heart\n"
            
        # Add other people if they exist
        other_people = [p for p in self.entities["PERSON"] if p != main_person][:2]  # Limit to 2 other people
        if other_people:
            people_list = " and ".join(other_people)
            verse += f"With {people_list} in their tale\n"
            verse += f"Each playing their important part"
        else:
            verse += f"Walking paths both new and old\n"
            verse += f"Each step a brand new start"
            
        return verse
        
    def _create_what_verse(self, summary):
        """Create a verse about what happened"""
        # Extract key sentences from summary
        sentences = re.split(r'[.!?]+', summary)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0][:2]  # Limit to 2 key sentences
        
        verse = f"Verse about WHAT:\n"
        
        if sentences:
            # Transform sentences into poetic lines
            for sentence in sentences:
                words = sentence.split()
                if len(words) > 8:
                    # Split into two lines if long
                    mid_point = len(words) // 2
                    verse += " ".join(words[:mid_point]) + "\n"
                    verse += " ".join(words[mid_point:]) + "\n"
                else:
                    verse += sentence + "\n"
        
        # Add a concluding line if verse is short
        if len(verse.split('\n')) < 4:
            verse += "A chapter in their life's great book\n"
            verse += "A memory that will remain"
            
        return verse
        
    def _create_where_verse(self, location):
        """Create a verse about where it happened"""
        verse = f"Verse about WHERE:\n"
        
        verse += f"In the realm of {location}\n"
        verse += f"Where stories come alive\n"
        
        # Add other locations if they exist
        other_locations = []
        if self.entities["LOC"]:
            other_locations.extend([l for l in self.entities["LOC"] if l != location])
        if self.entities["GPE"]:
            other_locations.extend([g for g in self.entities["GPE"] if g != location])
            
        other_locations = other_locations[:2]  # Limit to 2 other locations
        
        if other_locations:
            locations_text = " and ".join(other_locations)
            verse += f"From {location} to {locations_text}\n"
            verse += f"Their journey does survive"
        else:
            verse += f"In spaces both grand and small\n"
            verse += f"Their story comes to thrive"
            
        return verse
        
    def _create_emotion_verse(self, dominant_emotion, emotion_opposite):
        """Create a verse about emotional journey"""
        verse = f"Verse about FEELINGS:\n"
        
        verse += f"In waves of {dominant_emotion} they swam\n"
        verse += f"Through tides of {emotion_opposite} they flew\n"
        
        # Add emotional journey
        if len(self.emotion_history) > 3:
            emotions_sample = random.sample(self.emotion_history, min(3, len(self.emotion_history)))
            emotions_text = ", ".join(emotions_sample)
            verse += f"Feelings of {emotions_text}\n"
            verse += f"Painting their story anew"
        else:
            verse += f"Each feeling a color bright\n"
            verse += f"Painting their world anew"
            
        return verse
    
    def summarize_conversation(self):
        """Create a structured summary with lyrical presentation"""
        # Only summarize if there's meaningful content
        if len(self.story_segments) < 2:
            return "Not enough information shared to create a meaningful summary."
        
        # Generate the lyrical summary
        lyrical_summary = self.create_lyrical_summary()
        
        # Create emotional profile
        emotion_counter = defaultdict(int)
        for emotion in self.emotion_history:
            emotion_counter[emotion] += 1
        
        dominant_emotion = max(emotion_counter.items(), key=lambda x: x[1])[0] if emotion_counter else "neutral"
        
        # Format the full summary report
        report = f"=== LIFE STORY AS A SONG ===\n\n"
        report += lyrical_summary
        report += "\n\n=== ANALYSIS ===\n\n"
        
        # Add structured data about WHO
        report += "WHO: "
        if self.entities["PERSON"]:
            report += ", ".join(self.entities["PERSON"])
        else:
            report += "Unspecified individual"
        report += "\n\n"
        
        # Add structured data about WHERE
        report += "WHERE: "
        locations = []
        if self.entities["LOC"]:
            locations.extend(self.entities["LOC"])
        if self.entities["GPE"]:
            locations.extend(self.entities["GPE"])
            
        if locations:
            report += ", ".join(locations)
        else:
            report += "Unspecified location"
        report += "\n\n"
        
        # Add structured data about WHEN
        report += "WHEN: "
        times = []
        if self.entities["DATE"]:
            times.extend(self.entities["DATE"])
        if self.entities["TIME"]:
            times.extend(self.entities["TIME"])
            
        if times:
            report += ", ".join(times)
        else:
            report += "Unspecified time"
        report += "\n\n"
        
        # Add emotional journey
        report += "EMOTIONAL JOURNEY: " + ", ".join([emotion.title() for emotion in self.emotion_history[-5:]]) + "\n"
        report += f"Dominant emotion: {dominant_emotion.title()}\n\n"
        
        # Add key topics if available
        if self.mentioned_topics:
            report += "KEY TOPICS: " + ", ".join(self.mentioned_topics).title() + "\n\n"
            
        report += f"Conversation turns: {self.conversation_turns}"
        
        return report

# Initialize chatbot
chatbot = EmotionalStoryChatbot()

# Initial message
print("=" * 60)
print("🤖 Emotional Story Chatbot")
print("=" * 60)

# AI initiates the conversation first
opening_message = chatbot.start_conversation()
print(f"🤖 AI: {opening_message}")

print("(Type 'summarize' to get a summary of our conversation, or 'exit' to quit)")
print("=" * 60)

# Main conversation loop
while True:
    user_input = input("🧒 You: ")
    
    # Check for exit command
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("🤖 Thank you for chatting with me. Take care!")
        break
    
    # Check for summarize command
    if user_input.lower() == 'summarize':
        summary = chatbot.summarize_conversation()
        print("\n" + "=" * 60)
        print("🔍 CONVERSATION SUMMARY")
        print("=" * 60)
        print(summary)
        print("=" * 60 + "\n")
        continue
    
    # Process normal input
    emotion, emotion_score = chatbot.detect_emotion(user_input)
    print(f"🎭 Emotion detected: {emotion} ({emotion_score:.2f})")
    
    # Generate and display response
    response = chatbot.generate_response(user_input, emotion, emotion_score)
    print("🤖 AI:", response)
