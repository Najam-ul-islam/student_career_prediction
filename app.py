from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import json
from random import randint
app = Flask(__name__)
model = pickle.load(open("decision_tree_entropy_2.pkl", "rb"))


@app.route("/predict", methods=['POST'])
def predict():
    # Get the data from the POST request.
    if request.method == 'POST':
        operating_system = int(request.form['Percentage_in_Operating_Systems'])
        algo = int(request.form['Percentage_in_Algorithms'])
        programming = int(request.form['Percentage_in_Programming_Concepts'])
        se = int(request.form['Percentage_in_Software_Engineering'])
        cn = int(request.form['Percentage_in_Computer_Networks'])
        elect = int(request.form['Percentage_in_Electronics_Subjects'])
        computer = int(request.form['Percentage_in_Computer_Architecture'])
        maths = int(request.form['Percentage_in_Mathematics'])
        comm_skills = int(request.form['Percentage_in_Communication_skills'])
        hours_per_day = int(request.form['Hours_working_per_day'])
        logical = int(request.form['Logical_quotient_rating'])
        hackathon = int(request.form['hackathons'])
        coding_skills = int(request.form['coding_skills_rating'])
        public_speaking = int(request.form['public_speaking_points'])

        df = pd.DataFrame([[operating_system, algo, programming, se, cn, elect, computer, maths, comm_skills,
                            hours_per_day, logical, hackathon, coding_skills, public_speaking, ]],
                          index=['Input'], columns=['Percentage_in_Operating_Systems',
                                                    'Percentage_in_Algorithms', 'Percentage_in_Programming_Concepts',
                                                    'Percentage_in_Software_Engineering', 'Percentage_in_Computer_Networks',
                                                    'Percentage_in_Electronics_Subjects', 'Percentage_in_Computer_Architecture',
                                                    'Percentage_in_Mathematics', 'Percentage_in_Communication_skills',
                                                    'Hours_working_per_day', 'Logical_quotient_rating', 'hackathons',
                                                    'coding_skills_rating', 'public_speaking_points'])
        df = df.astype(float)
        prediction = model.predict(df)[0]
        original_input = {'Percentage_in_Operating_Systems': operating_system,
                          'Percentage_in_Algorithms': algo, 'Percentage_in_Programming_Concepts': programming,
                          'Percentage_in_Software_Engineering': se, 'Percentage_in_Computer_Networks': cn,
                          'Percentage_in_Electronics_Subjects': elect, 'Percentage_in_Computer_Architecture': computer,
                          'Percentage_in_Mathematics': maths, 'Percentage_in_Communication_skills': comm_skills,
                          'Hours_working_per_day': hours_per_day, 'Logical_quotient_rating': logical, 'hackathons': hackathon,
                          'coding_skills_rating': coding_skills, 'public_speaking_points': public_speaking, "Result": int(prediction+randint(1, 80))}

    return json.dumps(original_input)


if __name__ == "__main__":
    app.run(debug=True)
