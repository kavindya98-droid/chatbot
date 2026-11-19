from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()

book_agent_name = "Chatty Bookworm"

def chat(Message):
    llm = init_chat_model ("openai/gpt-oss-120b", model_provider= "groq", temperature=0,max_tokens=1024)
    book_desp = f"""
        Narnia is a mythical, magical world created by author C.S. Lewis in his famous seven-book fantasy series, The Chronicles of Narnia. It is a world filled with magic, talking animals, mythical beasts, and deep spiritual symbolism.

        Key Characteristics of Narnia
        Origins & Creation: Narnia was spoken and sung into existence by Aslan, the great and noble Lion who serves as the creator, protector, and ultimate authority of the realm.

        Inhabitants:

        Talking Beasts: Animals capable of speech, reason, and human-like emotions (e.g., Mr. and Mrs. Beaver, Reepicheep the Mouse).

        Mythological Creatures: Centaurs, fauns (such as Mr. Tumnus), dryads, naiads, dwarfs, giants, and minotaurs.

        Humans (Sons of Adam & Daughters of Eve): Humans do not naturally originate from Narnia; they enter from our world through magical portals (like the famous wardrobe) or are brought there by Aslan. High Kings and Queens of Narnia are traditionally humans.

        Geography & Notable Landmarks
        Narnia is both the name of the overall universe/world and the specific kingdom within it:

        Cair Paravel: The majestic castle located on the eastern coast, serving as the seat of the High Kings and Queens of Narnia.

        The Lantern Waste: A snowy woodland area near the western border where Lucy Pevensie first enters Narnia, marked by an improbable iron lamp-post glowing in the woods.

        The Great River: A major river winding through the heart of Narnia, featuring key locations like the Fords of Beruna.

        Neighboring Lands:

        Archenland: A mountainous realm directly to the south, friendly to Narnia.

        Calormen: A vast, arid, orientalist empire further south beyond the desert, often hostile toward Narnia.

        The Eastern Ocean: Dotted with mysterious islands leading toward Aslan's Country at the world's end.

        Time & Access
        Flow of Time: Time in Narnia runs independently of time in our world. Years or centuries can pass in Narnia while only seconds or minutes elapse on Earth.

        Portals: Access to Narnia occurs when Aslan wills it, often via magical objects or places (e.g., the magical Wardrobe, pictures hanging on a wall, or magic rings).

        Central Themes & Atmosphere
        Narnia is defined by its battle between good and evil, themes of redemption, sacrifice, royalty, and faith. The environment ranges from the harsh, perpetual winter under the rule of the White Witch to bright, enchanting summers restored by Aslan's presence.
        """

    messages = [
        SystemMessage(content=book_desp),
        HumanMessage(content=Message)
    ]

    response = llm.invoke(messages)
    print(response.content)

user_name = input("Enter Your name: ")
if user_name:

    for i in range(10):
        user_message = input("Enter your message: ")
        if user_message == "q":
            break

        if user_message:
            response = chat(user_message)
            print(response)

        else:
            print("Please Enter a Valid Message")
else:
    print("Please Enter a Valid Name")   