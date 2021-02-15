- Using python 3.6 or greater for cluster.py and email.ipynb

### cluster.py
The class takes in a set of data, and creates a K-Means model.
Helper functions for plotting and extracting features, generating common words and optimization of clusters also included.

### email.ipynb
I chose to use a K-Means clustering algorithm to better gauge the contents of the emails as a whole.

I decided to partition the email corpus into several categories. Emails containing phrases "fraud", "litigation", "bankrupt". Emails sent and recieved from June 2001 to August 2001 to see if any suspicious activity was mentioned prior to Skilling's departure from Enron.  I also checked the frequency with which Skilling sent or recieved messages from specific contacts.

