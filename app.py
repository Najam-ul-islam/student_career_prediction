from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import json

app = Flask(__name__)
model = pickle.load(open("decision_tree_new.pkl", "rb"))


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


@app.route("/predict", methods=['POST'])
def predict():
    # Get the data from the POST request.
    if request.method == 'POST':
        operating_system = request.form['Percentage_in_Operating_Systems']
        algo = request.form['Percentage_in_Algorithms']
        programming = request.form['Percentage_in_Programming_Concepts']
        se = request.form['Percentage_in_Software_Engineering']
        cn = request.form['Percentage_in_Computer_Networks']
        elect = request.form['Percentage_in_Electronics_Subjects']
        computer = request.form['Percentage_in_Computer_Architecture']
        maths = request.form['Percentage_in_Mathematics']
        comm_skills = request.form['Percentage_in_Communication_skills']
        hours_per_day = request.form['Hours_working_per_day']
        logical = request.form['Logical_quotient_rating']
        hackathon = request.form['hackathons']
        coding_skills = request.form['coding_skills_rating']
        public_speaking = request.form['public_speaking_points']
        long_time_work = request.form['can_work_long_time_before_system']
        self_learning = request.form['self-learning_capability']
        extra_courses = request.form['Extra-courses_did']
        talent_test = request.form['talenttests_taken']
        olympiads = request.form['olympiads']
        rw_skills = request.form['reading_and_writing_skills']
        memory = request.form['memory_capability_score']
        interested_subj = request.form['Interested_subjects']
        df = pd.DataFrame([[operating_system, algo, programming, se, cn, elect, computer, maths, comm_skills, hours_per_day, logical, hackathon, coding_skills, public_speaking, long_time_work, self_learning, extra_courses, talent_test, olympiads, rw_skills, memory, interested_subj]],
                          index=['Input'], dtype=int, columns=['Percentage_in_Operating_Systems', 'Percentage_in_Algorithms', 'Percentage_in_Programming_Concepts', 'Percentage_in_Software_Engineering', 'Percentage_in_Computer_Networks', 'Percentage_in_Electronics_Subjects', 'Percentage_in_Computer_Architecture', 'Percentage_in_Mathematics', 'Percentage_in_Communication_skills', 'Hours_working_per_day', 'Logical_quotient_rating', 'hackathons', 'coding_skills_rating', 'public_speaking_points', 'can_work_long_time_before_system', 'self-learning_capability', 'Extra-courses_did', 'talenttests_taken', 'olympiads', 'reading_and_writing_skills', 'memory_capability_score', 'Interested_subjects'])
        # df = df.astype(float)

        prediction = model.predict(df)[0]
        original_input = {'Percentage_in_Operating_Systems': operating_system,
                          'Percentage_in_Algorithms': algo, 'Percentage_in_Programming_Concepts': programming, 'Percentage_in_Software_Engineering': se, 'Percentage_in_Computer_Networks': cn, 'Percentage_in_Electronics_Subjects': elect, 'Percentage_in_Computer_Architecture': computer, 'Percentage_in_Mathematics': maths, 'Percentage_in_Communication_skills': comm_skills, 'Hours_working_per_day': hours_per_day, 'Logical_quotient_rating': logical, 'hackathons': hackathon, 'coding_skills_rating': coding_skills, 'public_speaking_points': public_speaking, 'can_work_long_time_before_system': long_time_work, 'self-learning_capability': self_learning, 'Extra-courses_did': extra_courses, 'talenttests_taken': talent_test, 'olympiads': olympiads, 'reading_and_writing_skills': rw_skills, 'memory_capability_score': memory, 'Interested_subjects': interested_subj, 'Prediction': prediction}
    # return jsonify(original_input)
    return json.dumps(original_input, cls=NpEncoder)


if __name__ == "__main__":
    app.run(debug=True)
