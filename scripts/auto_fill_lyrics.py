import os

lyrics_data = {
    "01": """Edelweiss edelweiss
Every morning you greet me
Small and white
Clean and bright
You look happy to meet me
Blossom of snow may you bloom and grow
Bloom and grow forever
Edelweiss edelweiss
Bless my homeland forever""",
    "02": """Doe a deer a female deer
Ray a drop of golden sun
Me a name I call myself
Far a long long way to run
Sew a needle pulling thread
La a note to follow sew
Tea a drink with jam and bread
That will bring us back to do oh oh oh""",
    "03": """Dashing through the snow
In a one horse open sleigh
O'er the fields we go
Laughing all the way
Bells on bob tails ring
Making spirits bright
What fun it is to laugh and sing
A sleighing song tonight
Oh jingle bells jingle bells
Jingle all the way
Oh what fun it is to ride
In a one horse open sleigh
Jingle bells jingle bells
Jingle all the way
Oh what fun it is to ride
In a one horse open sleigh""",
    "04": """Silent night holy night
All is calm all is bright
Round yon Virgin Mother and Child
Holy Infant so tender and mild
Sleep in heavenly peace
Sleep in heavenly peace""",
    "05": """It's late in the evening she's wondering what clothes to wear
She puts on her make-up and brushes her long blonde hair
And then she asks me Do I look all right
And I say Yes you look wonderful tonight
We go to a party and everyone turns to see
This beautiful lady that's walking around with me
And then she asks me Do you feel all right
And I say Yes I feel wonderful tonight""",
    "06": """Moon river wider than a mile
I'm crossing you in style some day
Oh dream maker you heart breaker
Wherever you're going I'm going your way
Two drifters off to see the world
There's such a lot of world to see
We're after the same rainbow's end
Waiting round the bend
My huckleberry friend
Moon river and me""",
    "07": """Listen to the rhythm of the falling rain
Telling me just what a fool I've been
I wish that it would go and let me cry in vain
And let me be alone again
The only girl I care about has gone away
Looking for a brand new start
But little does she know that when she left that day
Along with her she took my heart""",
    "08": """I have a dream a song to sing
To help me cope with anything
If you see the wonder of a fairy tale
You can take the future even if you fail
I believe in angels
Something good in everything I see
I believe in angels
When I know the time is right for me
I'll cross the stream I have a dream""",
    "09": """Love me tender love me sweet
Never let me go
You have made my life complete
And I love you so
Love me tender love me true
All my dreams fulfill
For my darling I love you
And I always will""",
    "10": """Twinkle twinkle little star
How I wonder what you are
Up above the world so high
Like a diamond in the sky
Twinkle twinkle little star
How I wonder what you are""",
    "11": """You are my sunshine my only sunshine
You make me happy when skies are gray
You'll never know dear how much I love you
Please don't take my sunshine away""",
    "12": """Greatness as you
Smallest as me
You show me what is deep as sea
A little love little kiss
A little hug little gift
All of little something these are our memories
You make me cry
Make me smile
Make me feel that love is true
You always stand by my side
I don't want to say goodbye""",
    "13": """Love in your eyes
Sitting silent by my side
Going on holding hands
Walking through the nights
Hold me up hold me tight
Lift me up to touch the sky
Teaching me to love with heart
Helping me open my mind
I can fly
I'm proud that I can fly
To give the best of mine
Till the end of the time
Believe me I can fly
I'm proud that I can fly
To give the best of mine
The heaven in the sky""",
    "14": """I'm sitting here in the boring room
It's just another rainy Sunday afternoon
I'm wasting my time I got nothing to do
I'm hanging around I'm waiting for you
But nothing ever happens and I wonder
I'm driving around in my car
I'm driving too fast I'm driving too far
I'd like to change my point of view
I feel so lonely I'm waiting for you
But nothing ever happens and I wonder""",
    "15": """There's a calm surrender to the rush of day
When the heat of a rolling wind can be turned away
An enchanted moment and it sees me through
It's enough for this restless warrior just to be with you
And can you feel the love tonight
It is where we are
It's enough for this wide-eyed wanderer
That we got this far""",
    "16": """I'm loving living every single day
But sometimes I feel so
I hope to find a little peace of mind
And I just want to know
And who can heal those tiny broken hearts
And what are we to be
Where is home on the milky way of stars
I dry my eyes again
In my dreams I am not so far away from home
What am I in a world so far away from home
All my life all the time so far away from home
Without you I will be so far away from home""",
    "17": """Goodbye to you my trusted friend
We've known each other since we were nine or ten
Together we've climbed hills and trees
Learned of love and ABC's
Skinned our hearts and skinned our knees
Goodbye my friend it's hard to die
When all the birds are singing in the sky
Now that the spring is in the air
Pretty girls are everywhere
Think of me and I'll be there""",
    "18": """I'm just a little bit caught in the middle
Life is a maze and love is a riddle
I don't know where to go I can't do it alone I've tried
And I don't know why
I'm just a little girl lost in the moment
I'm so scared but I don't show it
I can't figure it out
It's bringing me down I know
I've got to let it go
And just enjoy the show""",
    "19": """Some say love it is a river
That drowns the tender reed
Some say love it is a razor
That leaves your soul to bleed
Some say love it is a hunger
An endless aching need
I say love it is a flower
And you its only seed""",
    "20": """Oceans apart day after day
And I slowly go insane
I hear your voice on the line
But it doesn't stop the pain
If I see you next to never
How can we say forever
Wherever you go whatever you do
I will be right here waiting for you
Whatever it takes or how my heart breaks
I will be right here waiting for you"""
}

def main():
    lyrics_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "interim", "lyrics"))
    os.makedirs(lyrics_dir, exist_ok=True)
    
    print("Auto-filling lyrics for all 20 songs...")
    for song_id, lyrics in lyrics_data.items():
        out_file = os.path.join(lyrics_dir, f"{song_id}.txt")
        # Write clean lyrics (no punctuation, lowercase)
        # The preprocessing expects clean words
        clean_lyrics = lyrics.replace('\n', ' ').replace(',', '').replace('.', '').replace('?', '').replace("'", "").lower()
        # Just write as-is, the pipeline can tokenize it
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(lyrics.strip() + "\\n")
        print(f"Written: {song_id}.txt")
        
    print("\\nAll done! You can now run `python scripts/prepare_data.py`")

if __name__ == "__main__":
    main()
