// src/data/loadSamplePosts.js
import YAML from 'js-yaml';

async function loadPostsIndex() {
  const response = await fetch('/posts/index.yaml');
  const yamlText = await response.text();
  return YAML.load(yamlText);
}

async function loadPost(url) {
  const response = await fetch(`/posts/${url}/content.md`);
  const markdownText = await response.text();
  return markdownText;
}

export { loadPostsIndex, loadPost };
