# glm-simple-evals


## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.glm.ai/glm-reasoning-alg/glm-simple-evals.git
git branch -M main
git push -uf origin main
```

# Running the evals

For the zai SDK, you can run evals as follows:
```python
python3 evaluate.py 
--model_name glm-4.5 # or glm-4.5-air
--backbone zai 
--zai_api_key xxxxxx
--save_dir /save-results-path 
--tasks aime 
--proc_num 100 
--checker vllm
--checker_url xxxx
--auto_extract_answer 
--max_new_tokens 81920 
--stream
```


For [BigModel](https://docs.bigmodel.cn/api-reference/%E6%A8%A1%E5%9E%8B-api/%E5%AF%B9%E8%AF%9D%E8%A1%A5%E5%85%A8), you can run evals as follows:

```python
python3 evaluate.py 
--checker vllm 
--model_name glm-4.5 # or glm-4.5-air
--backbone openai 
--openai_base_url https://open.bigmodel.cn/api/paas/v4/ 
--openai_api_key xxxxxx 
--save_dir /results/save-path 
--tasks aime 
--proc_num 100 
--auto_extract_answer 
--max_new_tokens 81920 
--stream
```

For [SiliconFlow](https://docs.siliconflow.com/en/api-reference/chat-completions/chat-completions), you can run evals as follows:

```python
python3 evaluate.py 
--checker vllm 
--model_name z-ai/GLM-4.5 # or z-ai/GLM-4.5-Air
--backbone openai 
--openai_base_url https://api.siliconflow.com/v1/
--openai_api_key sk-xxxxxx 
--save_dir /results/save-path 
--tasks aime 
--proc_num 100 
--auto_extract_answer 
--max_new_tokens 81920 
--stream
```

