from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

load_dotenv()

health_agent_name = "Health Agent"

Embedding_model = "nvidia/nemotron-3.5-embed-1b:free"
chunk_size = 1000
chunk_overlap = 200
top_k = 2

vectorstores = None

def create_rag(doc_name):
  global vectorstores
  print("Starting RAG Creation...")
  reader = PdfReader(f"docs/{doc_name}")
  number_of_pages = len(reader.pages)
  text = ""

  for i in range(0, number_of_pages):
    text =  text + "\n\n" + reader.pages[i].extract_text()

  #split in to chunks
  text_splitter = RecursiveCharacterTextSplitter(
  chunk_size= chunk_size,
  chunk_overlap= chunk_overlap
  )

  chunks = text_splitter.split_text(text)

  #print(chunks)
  #number of chunks
  # print(len(chunks))

  # for i in range(1,len(chunks)):
  #     print(chunks[i],"\n\n\n")

  #openrouter
  # embeddings = OpenAIEmbeddings(
  #     model=Embedding_model,
  #     base_url="https://openrouter.ai/api/v1/",
  #     api_key = os.getenv("OPEN_ROUTER_API_KEY")
  # )

  embeddings = GoogleGenerativeAIEmbeddings(
      model="gemini-embedding-2-preview"
  )

  vectorstores = FAISS.from_texts(chunks, embeddings)
  print("RAG Created Successfully!...")

def chat(user_name, country, preferred_metric, food_preference, preferred_workout_type, height, weight):
    llm = init_chat_model("openai/gpt-oss-120b", model_provider="groq", temperature=0, max_tokens=1024)

    query = (
    f"User: {user_name}, Country: {country}, Metric: {preferred_metric}, "
    f"Food preference: {food_preference}, Workout preference: {preferred_workout_type}, "
    f"Height: {height}, Weight: {weight}"
    )

    docs = vectorstores.similarity_search(query, k=top_k)
    #docs = vectorstores.similarity_search(country,preferred_metric,food_preference,preferred_workout_type,height,weight,top_k=top_k)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # SYSTEM PROMPT: Contains all logic, regional rules, and output structuring
    bmi_calculator = f"""
        You are a friendly and encouraging health and wellness assistant {health_agent_name} that has 10 years of experience. 
        You are very good at calculating Body Mass Index (BMI) and providing tailored, foundational lifestyle guidance.

        When processing the user's details (Name, Country, Preferred Unit Metric, Food Preference, Preferred Workout Type, Height, and Weight):
        1. Identify the input values and confirm the preferred measurement system ({preferred_metric}) and user's location ({country}).
        2. Calculate the BMI internally using the correct formula based on the preferred metric:
           - Metric (cm/kg): BMI = weight (kg) / [height (m)]²
           - Imperial (inches/lbs): BMI = 703 × weight (lbs) / [height (in)]²
        3. Round the resulting BMI score to one decimal place.
        4. Categorize the BMI based on standard adult ranges:
           - Below 18.5: Underweight
           - 18.5 – 24.9: Normal weight
           - 25.0 – 29.9: Overweight
           - 30.0 or higher: Obesity

        <context>
        {context}
        </context>

        5. **Contextual Recommendations**: Based on the calculated BMI category and location ({country}):
           - **Diet Plan**: 
             * Filter and recommend food choices strictly matching the user's food preference ({food_preference}). If Vegetarian/Vegan, strictly exclude meat/fish/poultry and recommend plant-based protein options from the context.
             * **Local Ingredient Accessibility & Affordability**: Consider the user's country ({country}). Prioritize ingredients from the context that are widely available, culturally relevant, and affordable in that region. For instance, if imported staples like Greek yogurt, berries, or quinoa are expensive or hard to find in {country} (e.g., Sri Lanka), replace them with accessible local context equivalents (e.g., local curd, green gram, dhal, papaya, red rice).
           - **Exercise Plan**: Adapt the workout guidance from the context to incorporate or complement the user's preferred workout style ({preferred_workout_type}) safely for their BMI level.

        6. **Format the Output**: Present everything in a clean, clear, and personalized format addressing the user by name. Structure it like this:
           - Personalized greeting & summary of profile inputs (including country: {country}).
           - **BMI Score & Category:** Display *only* the final rounded BMI value and category (do not show the math equation).
           - **Tailored Diet Plan:** Nutrition tips tailored strictly to their dietary preference ({food_preference}) using locally accessible and affordable ingredients for {country}.
           - **Tailored Exercise Plan:** Workout tips incorporating their preferred style ({preferred_workout_type}).
           - **Disclaimer:** A clear note that BMI and these general tips are not a substitute for professional medical or nutritional advice.

        If any required information is missing, politely ask the user to provide the missing details.

        Use only the information and recommendations provided in the context. Give output in a healthy and respectful manner.        
    """

    messages = [
        SystemMessage(content=bmi_calculator),
        HumanMessage(
            content=f"Name: {user_name}, "
                    f"Country: {country}, "
                    f"Preferred Metric: {preferred_metric}, "
                    f"Food Preference: {food_preference}, "
                    f"Preferred Workout Type: {preferred_workout_type}, "
                    f"Height: {height}, "
                    f"Weight: {weight}"
        )
    ]

    response = llm.invoke(messages)
    return response.content

# user_name = input("Enter Your Name: ")
# if user_name:
#     country = input("Enter Your Country: ")
#     if country:
#       preferred_metric = input("Enter Your Preferred Metric (cm/kg or inches/lbs): ")
#       if preferred_metric:
#           food_preference = input("Enter Your Food Preference (veg/non-veg): ")
#           if food_preference:
#               preferred_workout_type = input("Enter Your Preferred Workout Type (gym/home/yoga/swimming): ")
#               if preferred_workout_type:
#                   height = input("Enter Your Height: ")
#                   if height:
#                       weight = input("Enter Your Weight: ")
#                       if weight:
#                           response = chat(
#                               user_name=user_name,
#                               country=country,
#                               preferred_metric=preferred_metric,
#                               food_preference=food_preference,
#                               preferred_workout_type=preferred_workout_type,
#                               height=height,
#                               weight=weight
#                           )
#                           print(response)
#                       else:
#                           print("Please Enter a Valid Weight")
#                   else:
#                       print("Please Enter a Valid Height")
#               else:
#                 print("Please Enter a Valid Workout Type")
#           else:
#             print("Please Enter a Valid Food Preference")
#       else:
#         print("Please Enter a Valid Metric")
#     else:
#       print("Please Enter a Valid Country")
# else:
#   print("Please Enter a Valid Name")   