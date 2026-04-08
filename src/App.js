// src/App.js
import React, { useEffect } from 'react';
import { HashRouter as Router, Route, Routes, useLocation } from 'react-router-dom';
import Header from './components/Header';
import BlogPosts from './components/BlogPosts';
import BlogPostPage from './components/BlogPostPage';
import About from './components/About';
import NotFoundPage from './components/404Page';
import Footer from './components/Footer';
import Projects from './components/Projects';
import ReactGA from 'react-ga4';

ReactGA.initialize("G-C02095RQYN");

function RouteTracker() {
  const location = useLocation();
  useEffect(() => {
    ReactGA.send({ hitType: "pageview", page: location.pathname });
  }, [location]);
  return null;
}

const App = () => {
  return (
    <Router>
      <RouteTracker />
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
