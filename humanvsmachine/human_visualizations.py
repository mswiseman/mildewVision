import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the two sheets into separate DataFrames
df_2_class = pd.read_excel('human_ vs_blackbird.xlsx', sheet_name='2_class_patches')
df_3_class = pd.read_excel('human_ vs_blackbird.xlsx', sheet_name='3_class_patches')

# Convert the 'correct' column to integers
df_2_class['correct_int'] = df_2_class['correct'].astype(int)
df_3_class['correct_int'] = df_3_class['correct'].astype(int)

# Add a new column 'class' to indicate the class
df_2_class['class'] = '2_class_patches'
df_3_class['class'] = '3_class_patches'

# Concatenate the two DataFrames
df_combined = pd.concat([df_2_class, df_3_class])

# Create a boxplot comparing the count of 'True' between the 2-class patches and 3-class patches
plt.figure(figsize=(10,7))
sns.boxplot(x='class', y='correct_int', data=df_combined)

# Set the plot title and labels
plt.title('Boxplot of Count of True Classifications for 2-class and 3-class Patches')
plt.xlabel('Class')
plt.ylabel('Count of True Classifications')

plt.savefig('human_class_performance_comparison.png')

# Create a pie chart of the distribution of actual classifications
class_distribution = df_2_class['actual'].value_counts()

plt.figure(figsize=(7,7))
plt.pie(class_distribution, labels=class_distribution.index, autopct='%1.1f%%', startangle=140)

# Set the plot title
plt.title('Distribution of Actual Classifications')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.savefig('class_distribution.png')

# Create a confusion matrix
confusion_matrix = pd.crosstab(df_2_class['actual'], df_2_class['response'], rownames=['Actual'], colnames=['Predicted'], margins = False)

# Plot the confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g')

# Set the plot title
plt.title('Confusion Matrix for Human Raters\' 2-Class Classifications')
plt.savefig('2_class_human_confusion_matrix.png')

# Create a confusion matrix
confusion_matrix = pd.crosstab(df_3_class['actual'], df_3_class['response'], rownames=['Actual'], colnames=['Predicted'], margins = False)

# Plot the confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g')

# Set the plot title
plt.title('Confusion Matrix for Human Raters\' 3-Class Classifications')
plt.savefig('3_class_human_confusion_matrix.png')



