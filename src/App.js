// src/App.js
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Header from './components/Header';
import BlogPosts from './components/BlogPosts';
import BlogPostPage from './components/BlogPostPage';
import About from './components/About';
import NotFoundPage from './components/404Page';
import Footer from './components/Footer';
import Projects from './components/Projects';
// import CV from './components/CV';
import ReactGA from 'react-ga';
const TRACKING_ID = "G-C02095RQYN";
ReactGA.initialize(TRACKING_ID);


const App = () => {
  return (
    <Router>
      <div className="App bg-gray-900 text-gray-100 min-h-screen">
        <Header />
        <Routes>
          <Route path="/" element={<BlogPosts />} />
          <Route path="/posts/" element={<BlogPosts />} />
          <Route path="/posts/:url" element={<BlogPostPage />} />
          <Route path="/about" element={<About />} />
          <Route path="/projects" element={<Projects />} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
