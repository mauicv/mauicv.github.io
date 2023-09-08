// src/components/BlogPostCard.js
import { Link } from 'react-router-dom';

function BlogPostCard({ post }) {
  return (
    <Link to={`/blogs/${post.url}`} className="block">
      <div className="bg-gray-800 p-4 my-4 rounded-lg flex items-center hover:bg-gray-700 transition duration-200">
        <img 
          src={post.image} 
          alt="Blog post" 
          className="w-1/4 rounded-lg mr-4" 
        />
        <div className="w-3/4">
          <h2 className="text-xl font-bold">{post.title}</h2>
          <div className="text-sm text-gray-400 flex justify-between mb-2">
            <span>{post.date}</span>
            <span>{post.topic}</span>
          </div>
          <p>{post.excerpt}</p>
        </div>
      </div>
    </Link>
  );
}

export default BlogPostCard;
