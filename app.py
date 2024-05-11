from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Define the route to render the HTML form
@app.route('/')
def index():
    return render_template("index.html")

# Define the route to handle the form submission
@app.route('/predict', methods=['POST'])
def predict():
    print("hiiii.html")
    if request.method == 'POST':
        # Retrieve form data
        test_input = request.form['testInput']
        gender = request.form['gender']
        nationality = request.form['nationality']
        place_of_birth = request.form['placeOfBirth']
        grade_id = request.form['gradeID']
        section_id = request.form['sectionID']
        topic = request.form['topic']
        semester = request.form['semester']
        relation = request.form['relation']
        raised_hands = request.form['raisedhands']
        visited_resources = request.form['visitedResources']
        announcements_view = request.form['announcementsView']
        discussion = request.form['discussion']
        parent_answering_survey = request.form['parentAnsweringSurvey']
        parent_college_satisfaction = request.form['parentCollegeSatisfaction']
        student_absence_days = request.form['studentAbsenceDays']
        
        # Process the data (you can perform your machine learning prediction here)
        prediction = process_data(test_input, gender, nationality, place_of_birth, grade_id, section_id, topic, semester, relation, raised_hands, visited_resources, announcements_view, discussion, parent_answering_survey, parent_college_satisfaction, student_absence_days)  # Example function to process data
        
        # Return the prediction as JSON
        return jsonify(prediction)


# Example function to process data
def process_data(test_input, gender, nationality, place_of_birth, grade_id, section_id, topic, semester, relation, raised_hands, visited_resources, announcements_view, discussion, parent_answering_survey, parent_college_satisfaction, student_absence_days):
    # Your processing logic here
    return test_input, gender, nationality, place_of_birth, grade_id, section_id, topic, semester, relation, raised_hands, visited_resources, announcements_view, discussion, parent_answering_survey, parent_college_satisfaction, student_absence_days

if __name__ == '__main__':
    app.run(debug=True)
