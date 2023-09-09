// src/components/BlogPostPage.js
import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { loadPostsIndex, loadPost} from '../data/loadData';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from "rehype-raw";
import 'katex/dist/katex.min.css';
import {Prism as SyntaxHighlighter} from 'react-syntax-highlighter'
import { dracula } from 'react-syntax-highlighter/dist/esm/styles/prism';


const components = {
  code({node, inline, className, children, ...props}) {
    const match = /language-(\w+)/.exec(className || '')
    return !inline && match ? (
      <SyntaxHighlighter
        {...props}
        children={String(children).replace(/\n$/, '')}
        style={dracula}
        language={match[1]}
        PreTag="div"
      />
    ) : (
      <code {...props} className={className}>
        {children}
      </code>
    )
  }
};


function BlogPostPage() {
  const { url } = useParams();
  const navigate = useNavigate();
  
  const [postMeta, setPostMeta] = useState([]);
  const [postContent, setPostContent] = useState([]);
  
  useEffect(() => {
    async function fetchData() {
      const data = await loadPostsIndex();
      const postMeta = data.find(p => p.url === url);
      if (!postMeta) {
        navigate('/404', { replace: true });
        return null;
      }
      const postContent = await loadPost(url);
      setPostMeta(postMeta);
      setPostContent(postContent)
    }
    fetchData();
  }, [navigate, url]);

  return (
    <div className="container mx-auto px-4 py-12 w-1/2">
      <h1 className="text-3xl font-bold mb-6">{postMeta.title}</h1>
      <img 
        src={postMeta.image} 
        alt={postMeta.title} 
        className="mb-4 rounded-lg" 
      />
      <ReactMarkdown 
          className="markdown-content"
          remarkPlugins={[remarkMath]}
          rehypePlugins={[rehypeKatex, rehypeRaw]}
          children={postContent}
          components={components}
        />
    </div>
  );
}

export default BlogPostPage;
