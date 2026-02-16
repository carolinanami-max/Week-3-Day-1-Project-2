class PromptTemplates:
    """Reusable prompt templates for personal branding LinkedIn content"""
    
    @staticmethod
    def linkedin_post():
        return """
        You are a personal branding expert creating authentic LinkedIn content.
        
        {context}
        
        Topic: {topic}
        Your Expertise: {expertise}
        Target Audience: {audience}
        
        REQUIREMENTS FOR AUTHENTIC PERSONAL BRAND:
        - Write in first-person perspective (I, me, my)
        - Share a personal insight, lesson learned, or real experience
        - Avoid corporate jargon and generic motivational quotes
        - Include a specific example or story from your work
        - End with a question to encourage comments
        - Keep it conversational, not salesy
        - Use short paragraphs for readability
        
        Create the LinkedIn post now:
        """
    
    @staticmethod
    def linkedin_carousel():
        return """
        Create a LinkedIn carousel post (slide-by-slide format) about:
        
        {context}
        
        Topic: {topic}
        Your Unique Perspective: {perspective}
        
        FORMAT:
        Slide 1 - Hook: [Attention-grabbing headline that reflects your personal brand]
        Slide 2 - Problem: [A challenge you personally faced or observed]
        Slide 3 - Insight: [What you learned from experience/context]
        Slide 4 - Action: [Practical advice your audience can use]
        Slide 5 - CTA: [Question to engage your network]
        
        Make each slide 1-2 sentences maximum. Be authentic, not promotional.
        """
    
    @staticmethod
    def thought_leadership():
        return """
        Write a thought leadership LinkedIn post that positions you as an expert.
        
        {context}
        
        Topic: {topic}
        Your Unique Angle: {angle}
        
        STRUCTURE:
        - Start with a controversial or bold statement (your real opinion)
        - Back it up with personal experience or data from context
        - Acknowledge the other side (show balanced thinking)
        - Share your conclusion and why it matters
        - Ask for others' perspectives
        
        Make it sound like YOU, not a generic AI. Be specific and opinionated.
        """
    
    @staticmethod
    def story_post():
        return """
        Turn this experience into an engaging LinkedIn story:
        
        {context}
        
        Topic: {topic}
        The Lesson: {lesson}
        
        STORY FRAMEWORK:
        - Setup: What happened before? (short)
        - Conflict: What challenge did you face? (specific details)
        - Resolution: How did you handle it? (your actions)
        - Takeaway: What did you learn? (value for audience)
        
        Write conversationally. Use "I" statements. Show vulnerability if appropriate.
        """
    
    @staticmethod
    def engagement_post():
        return """
        Create a high-engagement LinkedIn post that sparks conversation.
        
        {context}
        
        Topic: {topic}
        Question to Ask: {question}
        
        REQUIREMENTS:
        - Start with a relatable statement or observation
        - Share your honest take (even if slightly controversial)
        - Ask the community for their experience
        - Reply to comments to boost engagement
        
        Write the post now:
        """
    
    @staticmethod
    def uniqueness_check():
        return """
        Compare this generic AI content with our personal brand version:
        
        GENERIC LINKEDIN POST:
        {generic}
        
        OUR PERSONAL BRAND POST:
        {ours}
        
        Analyze the differences:
        1. Authenticity & personal voice:
        2. Specific examples vs generic advice:
        3. Engagement potential:
        """