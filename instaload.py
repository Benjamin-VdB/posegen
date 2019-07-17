# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:01:30 2019

@author: ben.vdb
"""

import instaloader

# Get instance
L = instaloader.Instaloader(sleep=True, quiet=False, user_agent=None, dirname_pattern=None
                            , filename_pattern=None, download_pictures=False, download_videos=True
                            , download_video_thumbnails=False, download_geotags=False
                            , download_comments=False, save_metadata=False, compress_json=False
                            , post_metadata_txt_pattern=None, storyitem_metadata_txt_pattern=None
                            , max_connection_attempts=3, commit_mode=False)

# Optionally, login or load session
#L.login("vandenbenj", "xxxxx11")        # (login)
#L.interactive_login("vandenbenj")

# Download from hashtag
#for post in L.get_hashtag_posts('poletricks'):
#    # post is an instance of instaloader.Post
#    L.download_post(post, target='#poletricks')
                    
                    
# hashtags: poledancenation
# tricks = [#pdspatchcock]
#L.download_hashtag('cat', max_count=10, post_filter=is_video)


user = 'nicocasa' # 'sarahscottpole' # phoenixkazree bridgetmonro
    
# Download from user
profile = instaloader.Profile.from_username(L.context, user)
i=0
for post in profile.get_posts():
    if i <500:
        print(i)
        L.download_post(post, target=profile.username)
        i+=1
# top 100 from user
#profile = instaloader.Profile.from_username(L.context, user)
#posts_sorted_by_likes = sorted(profile.get_posts(), key = lambda p: p.likes + p.comments)
#
#for post in islice(posts_sorted_by_likes, 100):
#    L.download_post(post, target=profile.username)