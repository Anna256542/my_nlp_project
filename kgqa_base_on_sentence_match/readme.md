基于知识图谱的问答系统（knowledge graph-based question answering system）
在运行代码之前需要先安装neo4j，具体见网盘
具体步骤如下：
1.首先需要安装jdk-14_windows-x64_bin
2.之后解压neo4j-community-4.4.8-windows文件夹
3.打开路径下的neo4j.bat文件。neo4j-community-4.4.8-windows\neo4j-community-4.4.8\bin
4.在cmd中的bin文件夹下输入：.\neo4j.bat console
5.稍等一会出现一个starded(说明已经在本地开启数据库)，窗口不要关闭
6.地址栏输入：http://127.0.0.1:7474/browser/
7.运行build_graph.py文件，将数据导入数据库中
8.运行graph_qa_base_on_sentence_match.py文件
