- Using python 3.6 or greater for cluster.py and email.ipynb

### cluster.py
I decided to create a class to create K-Means clustering model for a given set of data.  This allowed me to quickly test the clustering algorithm on different subsets of the email data.
The class takes in a set of data, and creates a K-Means model.
Helper functions for plotting and extracting features, generating common words and optimization of clusters also included.

### email.ipynb

After recognizing that the content/recipients of the emails in this dataset pertained to Enron and specifically Jeff Skilling, I did some background research on the company and its high level executives so I would be more familiar with this dataset's contents.

Because Enron was a high profile case in fraud, my initial reaction was to see if I could identify people or emails that related to Enron's criminal dealings.

I chose to use a K-Means clustering algorithm to better gauge the contents of Jeff Skilling's emails as a whole.

I decided to partition the email corpus into several categories. Emails containing phrases "fraud", "litigation", "bankrupt". Emails sent and recieved from June 2001 to August 2001 to see if any suspicious activity was mentioned prior to Skilling's departure from Enron.  I also checked the frequency with which Skilling sent or recieved messages from specific contacts.

##### Key Phrases
- The litigation filter provided the most interesting insights out of the three phrases searched. Top features in this model showed words directly related to the investigations and criminal activity. "Brownell" and "FERC" were commonly used in one cluster, along with "suspended", "lawsuit" and other phrases linked to criminal proceedings

##### Key Time Frame
- Searching the couple months prior to Skilling's resignation showed a high frequency of the words "assure" and "concern" along with mentions of Kenneth Lay.

##### Most Messaged Contacts
- Sherri Sera was a clear first in number of direct correspondences, Kenneth Lay was second in frequency.

###### Key Contact Emails
- It looks like the most interesting features from Sherri and Kenneth's emails are regarding meetings and conferences, with some discussion of executive leadership


#### Anomalies
- I noticed some emails that appeared to be from solicitors or spam, which did not have relevance to any investigation or fraud


### Next Steps
- To proceed from here I would look at combinations of these subsets to see if there are higher concentrations of what seems to be fraudulent activity. For example, searching key phrases within the key time frame.



