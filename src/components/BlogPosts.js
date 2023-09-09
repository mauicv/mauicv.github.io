// src/components/BlogPosts.js
import React, { useState, useEffect } from 'react';
import BlogPostCard from './BlogPostCard';
import { loadPostsIndex } from '../data/loadData';

function BlogPosts() {
  const [posts, setPosts] = useState([]);

  useEffect(() => {
    async function fetchData() {
      const data = await loadPostsIndex();
      setPosts(data.reverse());
    }
    fetchData();
  }, []);

  return (
    <div className="container mx-auto px-4 py-12 md:w-1/2">
      <h1 className="text-3xl font-bold mb-6">Blog Posts</h1>
      {posts.map(post => (
        <BlogPostCard key={post.id} post={post} />
      ))}
    </div>
  );
}

export default BlogPosts;
