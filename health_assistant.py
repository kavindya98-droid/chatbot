from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

health_agent_name = "Health Agent"

context = """
Plans by BMI Category (with Regional & Ingredient Guidelines):

1. Underweight (BMI < 18.5)
Context: The focus is on a healthy caloric surplus combined with progressive resistance training to gain lean muscle mass rather than just body fat.
Diet Plan Focus:
- Caloric Intake: Aim for a slight energy surplus (+300 to +500 calories/day above maintenance).
- Nutrient-Dense Foods: Incorporate healthy fats and high-protein foods. 
- Non-vegetarian: Lean meats, eggs, full-fat dairy (e.g., full-fat milk, whole curd/yogurt, cheese).
- Vegetarian/Vegan: Tofu, tempeh, lentils, dhal, beans, peanuts/peanut butter, seed butters, nuts, avocados, coconut, oats, and local plant milks.
- Regional Ingredient Alternatives:
  - Western/Europe/Americas: Greek yogurt, almond butter, avocados, whole-grain oats, olive oil.
  - South Asia (e.g., Sri Lanka, India): Buffalo/cow curd, peanuts, sesame seeds, coconut fats, chickpea flour (besan), dhal/lentils, eggs, local fish.
- Frequent Meals: Eat 3 main meals and 2–3 dense snacks throughout the day.
- Hydration Tip: Drink fluids between meals rather than right before to avoid premature fullness.
Exercise Plan Focus:
- Strength Training (Primary): Focus on compound resistance exercises (squats, deadlifts, overhead presses, rows) 3–4 days per week with moderate-to-heavy weights.
- Cardio (Minimal): Keep steady-state cardio light (15–20 minutes, 2 times a week) to protect heart health without burning excessive calories.

2. Normal Weight (BMI 18.5 – 24.9)
Context: The goal here is maintaining overall health, improving metabolic fitness, and optimizing body composition (muscle-to-fat ratio).
Diet Plan Focus:
- Caloric Intake: Eat at caloric maintenance.
- Balanced Macro Split:
  - Non-Vegetarian Proteins: Chicken, fish, eggs, dairy/curd.
  - Vegetarian/Vegan Proteins: Tofu, edamame, lentils/dhal, chickpeas, green gram (mung beans), seitan, plant/local yogurt.
  - Carbohydrates: Complex carbs like brown rice, red rice, quinoa, sweet potatoes, cassava/yams, and whole grains.
  - Healthy Fats: Olive oil, sesame oil, mustard oil, seeds, peanuts, nuts, and fatty fish (for non-vegetarians).
  - Whole Foods: Focus on rich micronutrients and minimal processed foods.
- Regional Ingredient Alternatives:
  - Western/Europe/Americas: Greek yogurt, quinoa, salmon, berries, olive oil.
  - South Asia (e.g., Sri Lanka, India): Plain local curd, red/brown rice, green gram, local white fish (e.g., tuna, mackerel), spinach/gotukola/local greens.
Exercise Plan Focus:
- Mixed Routine: Combine 3 days of resistance training (for strength and tone) with 2–3 days of moderate aerobic exercise (jogging, cycling, swimming, or sports).
- Flexibility & Mobility: Include yoga or dynamic stretching sessions twice a week.

3. Overweight (BMI 25.0 – 29.9)
Context: The goal is achieving a steady, sustainable caloric deficit while preserving muscle tissue and building physical endurance.
Diet Plan Focus:
- Caloric Intake: Maintain a mild caloric deficit (-300 to -500 calories/day).
- High-Protein & Fiber Focus: High protein keeps you full and protects lean muscle mass. Fibrous vegetables add volume to meals with few calories.
  - Non-Vegetarian: Lean poultry, local white fish, egg whites.
  - Vegetarian/Vegan: Tofu, lentils/dhal, black beans, green gram, lupini beans, plant-based protein powders.
  - Vegetables: Broccoli, spinach, bell peppers, leafy greens (gotukola, mukunuwenna, kale, etc.), cucumbers, cabbage.
- Portion Control: Mindful portioning of dense carbohydrates (rice, bread, potatoes) and added fats/oils.
- Regional Ingredient Alternatives:
  - Western/Europe/Americas: Low-fat Greek yogurt, turkey breast, broccoli, berries.
  - South Asia (e.g., Sri Lanka, India): Low-fat local curd, boiled chickpea/green gram, egg white omelets, local fresh vegetables, red rice in controlled portions.
- Hydration: Drink 2.5–3 liters of water daily to support digestion and satiety.
Exercise Plan Focus:
- Cardio: 150–200 minutes per week of moderate-intensity cardio (brisk walking, cycling, rowing, or ellipticals).
- Strength Training: Full-body resistance training 3 days a week to retain muscle while losing fat.

4. Obesity (BMI 30.0+)
Context: The priority is low-impact, sustainable movement to protect joints while implementing structured nutrition for gradual weight reduction.
Diet Plan Focus:
- Caloric Intake: Structured caloric deficit (-500 to -750 calories/day) focused on long-term sustainability rather than rapid restriction.
- Whole-Food Priority: Swap refined carbohydrates and ultra-processed foods for whole grains (red/brown rice, whole wheat), plant/lean proteins, and low-glycemic fruits/vegetables.
- Liquid Calories: Cut out sugar-sweetened beverages, fruit juices, and sodas in favor of water, sparkling water, or plain herbal tea.
- Regional Ingredient Alternatives:
  - Western/Europe/Americas: Low-fat cottage cheese/Greek yogurt, oats, berries, grilled chicken breast.
  - South Asia (e.g., Sri Lanka, India): Plain curd, boiled dhal/pulses, local leafy green salads/sambols (minimal oil), papaya, guava, local fish curries made with light coconut milk or water base.
Exercise Plan Focus:
- Low-Impact Movement (Primary): Start with joint-friendly cardio like swimming, water aerobics, stationary cycling, or brisk walking (20–30 minutes daily).
- Bodyweight & Band Resistance: Begin strength work using seated exercises, resistance bands, or light dumbbells 2–3 days per week to build foundational strength safely.
"""

def chat(user_name, country, preferred_metric, food_preference, preferred_workout_type, height, weight):
    llm = init_chat_model("openai/gpt-oss-120b", model_provider="groq", temperature=0, max_tokens=1024)
    
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
#     preferred_metric = input("Enter Your Preferred Metric (cm/kg or inches/lbs): ")
#     if preferred_metric:
#         food_preference = input("Enter Your Food Preference (veg/non-veg): ")
#         if food_preference:
#             preferred_workout_type = input("Enter Your Preferred Workout Type (gmoym/home/yoga/swimming): ")
#             if preferred_workout_type:
#                 height = input("Enter Your Height: ")
#                 if height:
#                     weight = input("Enter Your Weight: ")
#                     if weight:
#                         response = chat(
#                             user_name=user_name,
#                             preferred_metric=preferred_metric,
#                             food_preference=food_preference,
#                             preferred_workout_type=preferred_workout_type,
#                             height=height,
#                             weight=weight
#                         )
#                         print(response)
#                     else:
#                         print("Please Enter a Valid Weight")
#                 else:
#                     print("Please Enter a Valid Height")
#             else:
#                 print("Please Enter a Valid Workout Type")
#         else:
#             print("Please Enter a Valid Food Preference")
#     else:
#         print("Please Enter a Valid Metric")
# else:
#     print("Please Enter a Valid Name")   