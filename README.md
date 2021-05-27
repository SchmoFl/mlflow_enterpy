# mlflow_enterpy

## Technische Voraussetzungen

### Betriebssystem
Grundsätzlich sollten python und mlflow unter jedem Betriebssystem funktionieren. Allerdings ist die Funktionalität von mlflow unter Windows nicht von den Entwicklern
gewährleistet. Vorzugsweise sind also macOS oder Linux zu benutzen. Möglicherweise funktioniert es aber auch unter Windows.

### Software
* *python*: https://www.python.org/downloads/
* ggf. **git**: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git (zum Download des Repos)
* **mlflow**: https://www.mlflow.org/docs/latest/quickstart.html 
* optional (Voraussetzung für registry) *sqlite*: https://www.sqlite.org/download.html (oder via pip); alternativ *mysql* oder *postgresql*
* optional: *graphviz*: via pip

### Python Libraries

Folgende python libraries sollten via pip (oder conda) installiert werden:

* *pandas*
* *numpy*
* *scikit-learn*/*sklearn*
* *mlflow*
* *seaborn*
* *matplotlib*
* *requests*
* evtl. *keras*
* evtl. *tensorflow*
* optional: *graphviz* 

Um die erfolgreiche Installation von *mlflow* sicherzustellen, sollte der Kommandozeilenbefehl **mlflow ui** ausgeführt werden. Danach sollte unter *http://localhost:5000* die *mlflow* UI erreichbar sein. 
